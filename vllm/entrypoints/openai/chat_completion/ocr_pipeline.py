# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import io
import json
import re
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.logger import init_logger
from vllm.multimodal.utils import encode_image_url

logger = init_logger(__name__)

_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,(.+)$", re.DOTALL)
_PDF_DATA_URL_RE = re.compile(
    r"^data:application/pdf(?:;[^,]*)?;base64,(.+)$", re.DOTALL | re.IGNORECASE
)
_DISPLAY_BRACKET_SEP_RE = re.compile(re.escape(r"\]") + r"\s*" + re.escape(r"\["))
_INLINE_LATEX_RE = re.compile(r"\\\((.+?)\\\)")
_PDF_SCALE = 2.0

_ELEMENT_CONFIG = {
    "text": ("Text Recognition:", 4096, "text"),
    "content": ("Text Recognition:", 4096, "text"),
    "abstract": ("Text Recognition:", 4096, "text"),
    "reference": ("Text Recognition:", 2048, "text"),
    "reference_content": ("Text Recognition:", 2048, "text"),
    "vertical_text": ("Text Recognition:", 2048, "text"),
    "vision_footnote": ("Text Recognition:", 1024, "text"),
    "algorithm": ("Text Recognition:", 4096, "text"),
    "doc_title": ("Text Recognition:", 512, "text"),
    "paragraph_title": ("Text Recognition:", 512, "text"),
    "figure_title": ("Text Recognition:", 512, "text"),
    "table": ("Table Recognition:", 8192, "html"),
    "display_formula": ("Formula Recognition:", 2048, "latex"),
    "inline_formula": ("Text Recognition:", 1024, "latex"),
    "formula_number": ("Text Recognition:", 256, "text"),
    "header": ("Text Recognition:", 512, "text"),
    "footer": ("Text Recognition:", 512, "text"),
    "footnote": ("Text Recognition:", 1024, "text"),
    "aside_text": ("Text Recognition:", 1024, "text"),
    "number": ("Text Recognition:", 256, "text"),
}
_SKIP_LABELS = {"image", "header_image", "footer_image", "chart", "seal"}
_IGNORE_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}
_LABEL_MAP = {"formula": "display_formula"}

# PaddleOCR-VL-1.6 PP-DocLayoutV3 recipe (OmniDocBench v1.6 layout cache).
_LAYOUT_MODEL_NAME = "PP-DocLayoutV3"
_LAYOUT_THRESHOLD = 0.3
_LAYOUT_SHAPE_MODE = "auto"
_LAYOUT_PARAMS_FILENAME = "inference.pdiparams"
_MERGE_BBOXES_MODE = {
    3: "large",  # chart
    5: "large",  # display_formula
    6: "large",  # doc_title
    15: "large",  # inline_formula
    17: "large",  # paragraph_title
}
_NON_MERGE_LABELS = [
    "image",
    "header_image",
    "footer_image",
    "chart",
    "seal",
    "table",
]


def _to_paddle_device(device: str) -> str:
    device = (device or "").strip().lower()
    if not device or device.startswith("cpu"):
        return "cpu"
    if device.startswith(("cuda", "gpu")):
        _, _, index = device.partition(":")
        return f"gpu:{index if index.isdigit() else '0'}"
    return device


@dataclass
class OCRBlock:
    label: str
    order: int
    bbox: tuple[int, int, int, int]
    image: Image.Image
    content: str = ""


@dataclass
class OCRDocument:
    image: Image.Image | None = None
    pdf_data: bytes | None = None


class PPDocLayoutService:
    """PP-DocLayoutV3 through PaddleX plus PaddleOCR-VL box filtering,
    polygon-masked cropping, and cross-column text merging.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_crops: int = 64):
        self.model_path = model_path
        self.device = _to_paddle_device(device)
        self.max_crops = max_crops
        self._model: Any | None = None
        self._crop_by_boxes: Any | None = None
        self._filter_overlap_boxes: Any | None = None
        self._merge_blocks: Any | None = None
        self._lock = asyncio.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import importlib

            from paddlex import create_model

            crop_mod = importlib.import_module(
                "paddlex.inference.pipelines.components.common.crop_image_regions"
            )
            vl_utils = importlib.import_module(
                "paddlex.inference.pipelines.paddleocr_vl.uilts"
            )
        except Exception as exc:
            raise RuntimeError(
                "OCR layout requires paddlepaddle and paddlex "
                "(Paddle inference PP-DocLayoutV3). Original error: %s" % exc
            ) from exc

        params = Path(self.model_path) / _LAYOUT_PARAMS_FILENAME
        if not params.is_file():
            raise FileNotFoundError(
                "ocr_layout_model must be a Paddle inference dir "
                f"(missing {_LAYOUT_PARAMS_FILENAME}): {self.model_path}"
            )

        logger.info(
            "Loading PP-DocLayoutV3 from %s on %s", self.model_path, self.device
        )
        self._model = create_model(
            model_name=_LAYOUT_MODEL_NAME,
            model_dir=self.model_path,
            device=self.device,
            threshold=_LAYOUT_THRESHOLD,
            layout_nms=True,
            layout_merge_bboxes_mode=dict(_MERGE_BBOXES_MODE),
        )
        self._crop_by_boxes = crop_mod.CropByBoxes()
        self._filter_overlap_boxes = vl_utils.filter_overlap_boxes
        self._merge_blocks = vl_utils.merge_blocks

    def _boxes_from_predict(self, raw_boxes: list[dict[str, Any]]) -> list[dict]:
        boxes: list[dict] = []
        for box in raw_boxes:
            item = {
                "cls_id": int(box["cls_id"]),
                "label": box["label"],
                "score": float(box["score"]),
                "coordinate": [float(v) for v in box["coordinate"]],
            }
            polygon = box.get("polygon_points")
            if polygon is not None and len(np.asarray(polygon)):
                item["polygon_points"] = np.asarray(polygon, dtype=np.float64).reshape(
                    -1, 2
                )
            boxes.append(item)
        return boxes

    def _parse(self, image: Image.Image) -> list[OCRBlock]:
        self._load()
        assert self._model is not None
        assert self._crop_by_boxes is not None
        assert self._filter_overlap_boxes is not None
        assert self._merge_blocks is not None

        # Feed RGB as-is; the OmniDocBench v1.6 cache used this colorspace.
        rgb = np.asarray(image.convert("RGB"))
        results = list(
            self._model.predict(
                rgb,
                batch_size=1,
                layout_shape_mode=_LAYOUT_SHAPE_MODE,
                filter_overlap_boxes=False,
            )
        )
        raw_boxes = list(results[0].get("boxes", [])) if results else []
        boxes = self._boxes_from_predict(raw_boxes)
        if boxes:
            boxes = self._filter_overlap_boxes({"boxes": boxes}, _LAYOUT_SHAPE_MODE)[
                "boxes"
            ]
        cropped = self._crop_by_boxes(rgb, boxes, _LAYOUT_SHAPE_MODE) if boxes else []
        merged = self._merge_blocks(
            cropped,
            non_merge_labels=_NON_MERGE_LABELS,
            layout_shape_mode=_LAYOUT_SHAPE_MODE,
        )

        blocks: list[OCRBlock] = []
        for idx, block in enumerate(merged):
            crop = block.get("img")
            if crop is None:
                continue
            if self.max_crops > 0 and len(blocks) >= self.max_crops:
                logger.warning("Truncated OCR layout to %d blocks", self.max_crops)
                break
            raw_label = str(block["label"])
            x1, y1, x2, y2 = (int(round(float(v))) for v in block["box"])
            blocks.append(
                OCRBlock(
                    label=_LABEL_MAP.get(raw_label, raw_label),
                    order=idx,
                    bbox=(x1, y1, x2, y2),
                    image=Image.fromarray(crop),
                )
            )
        return blocks

    async def parse(self, image: Image.Image) -> list[OCRBlock]:
        async with self._lock:
            return await asyncio.to_thread(self._parse, image)


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _decode_base64(value: str, error: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError(error) from exc


def _extract_document(request: ChatCompletionRequest) -> OCRDocument | None:
    documents: list[OCRDocument] = []
    for message in request.messages:
        content = _get_field(message, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = _get_field(part, "type")
            if part_type == "image_url":
                image_url = _get_field(part, "image_url")
                url = _get_field(image_url, "url")
                if not isinstance(url, str):
                    continue
                match = _DATA_URL_RE.match(url)
                if match is None:
                    raise ValueError("OCR pipeline only accepts base64 image data URLs")
                raw = _decode_base64(match.group(1), "Invalid base64 image data URL")
                try:
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:
                    raise ValueError("Invalid base64 image data URL") from exc
                documents.append(OCRDocument(image=image))
            elif part_type == "file":
                file = _get_field(part, "file")
                file_data = _get_field(file, "file_data")
                filename = _get_field(file, "filename")
                if not isinstance(file_data, str):
                    if _get_field(file, "file_id"):
                        raise ValueError("OCR pipeline does not support file_id inputs")
                    raise ValueError("PDF input requires file.file_data")
                match = _PDF_DATA_URL_RE.match(file_data)
                encoded = match.group(1) if match is not None else file_data
                if (
                    match is None
                    and isinstance(filename, str)
                    and not filename.lower().endswith(".pdf")
                ):
                    raise ValueError("OCR pipeline only accepts PDF file inputs")
                documents.append(
                    OCRDocument(
                        pdf_data=_decode_base64(encoded, "Invalid base64 PDF file data")
                    )
                )
    if not documents:
        return None
    if len(documents) != 1:
        for document in documents:
            if document.image is not None:
                document.image.close()
        raise ValueError("OCR pipeline accepts exactly one image or PDF per request")
    return documents[0]


def _has_document(request: ChatCompletionRequest) -> bool:
    for message in request.messages:
        content = _get_field(message, "content")
        if isinstance(content, list) and any(
            _get_field(part, "type") in {"image_url", "file"} for part in content
        ):
            return True
    return False


def _open_pdf(pdf_data: bytes):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF") from exc

    try:
        document = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as exc:
        raise ValueError("Invalid PDF file data") from exc
    if document.needs_pass:
        document.close()
        raise ValueError("Password-protected PDF files are not supported")
    return document


def _get_pdf_page_count(pdf_data: bytes) -> int:
    document = _open_pdf(pdf_data)
    try:
        if document.page_count < 1:
            raise ValueError("PDF file has no pages")
        return document.page_count
    finally:
        document.close()


def _render_pdf_page(pdf_data: bytes, page_index: int) -> Image.Image:
    import fitz

    document = _open_pdf(pdf_data)
    try:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(_PDF_SCALE, _PDF_SCALE),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception as exc:
        raise ValueError(f"Failed to render PDF page {page_index + 1}") from exc
    finally:
        document.close()


def _truncate_repetitive_content(content: str, min_count: int) -> str:
    if len(content) < min_count:
        return content
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) >= 10:
        most_common, count = Counter(lines).most_common(1)[0]
        if count >= 10 and count / len(lines) >= 0.8:
            return most_common
    return content


def _postprocess_content(content: str, output_format: str, label: str) -> str:
    content = _truncate_repetitive_content(
        content.strip(), 5000 if label == "table" else 50
    )
    if output_format == "latex":
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if content.startswith(left) and content.endswith(right):
                content = content[len(left) : -len(right)].strip()
        if label == "display_formula":
            content = _DISPLAY_BRACKET_SEP_RE.sub("$$   $$", content)
            content = content.replace(r"\[", "").replace(r"\]", "")
    elif output_format == "html":
        if content.startswith("```html"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if label == "table":
            content += "</table>" * max(
                0, content.count("<table") - content.count("</table>")
            )
    if label != "display_formula":
        content = _INLINE_LATEX_RE.sub(
            lambda match: f"${match.group(1).strip()}$", content
        )
    content = re.sub(r"[ \t]+", " ", content)
    return re.sub(r"\n{3,}", "\n\n", content)


def _assemble_markdown(blocks: list[OCRBlock]) -> str:
    parts: list[str] = []
    for block in sorted(blocks, key=lambda item: item.order):
        label = block.label
        content = block.content
        if label in _IGNORE_LABELS:
            continue
        if label in {"image", "header_image", "footer_image"}:
            parts.append("![image]()")
        elif label in {"chart", "seal"}:
            parts.append(content or "![image]()")
        elif not content:
            continue
        elif label == "doc_title":
            parts.append(f"# {content}")
        elif label == "paragraph_title":
            parts.append(f"## {content}")
        elif label == "figure_title":
            parts.append(f"*{content}*")
        elif label == "display_formula":
            parts.append(f"$$\n{content}\n$$")
        elif label == "inline_formula":
            parts.append(f"${content}$")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _page_failure_marker(page_number: int, message: str) -> str:
    message = re.sub(r"\s+", " ", message).replace("--", "- -").strip()
    return f"<!-- Page {page_number} failed: {message[:500]} -->"


class OCRPipelineServing:
    def __init__(
        self,
        batch_serving: OpenAIServingChatBatch,
        layout_model: str,
        layout_device: str,
        max_crops: int,
        max_tokens: int,
        max_slice_nums: int,
        max_pdf_pages: int,
    ):
        if max_pdf_pages < 0:
            raise ValueError("ocr_max_pdf_pages must not be negative")
        self.batch_serving = batch_serving
        self.layout = PPDocLayoutService(layout_model, layout_device, max_crops)
        self.max_tokens = max_tokens
        self.max_slice_nums = max_slice_nums
        self.max_pdf_pages = max_pdf_pages

    def can_handle(self, request: ChatCompletionRequest) -> bool:
        return _has_document(request)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Any
    ) -> ChatCompletionResponse | ErrorResponse | AsyncGenerator[str, None]:
        try:
            document = _extract_document(request)
            assert document is not None
            if document.pdf_data is None:
                assert document.image is not None
                page_count = 1
            else:
                page_count = await asyncio.to_thread(
                    _get_pdf_page_count, document.pdf_data
                )
                if self.max_pdf_pages > 0 and page_count > self.max_pdf_pages:
                    raise ValueError(
                        f"PDF has {page_count} pages; maximum is {self.max_pdf_pages}"
                    )
        except ValueError as exc:
            return self.batch_serving.create_error_response(str(exc))
        except Exception as exc:
            logger.exception("OCR document loading failed")
            return self.batch_serving.create_error_response(
                f"OCR document loading failed: {exc}",
                err_type="InternalServerError",
                status_code=500,
            )

        page_markdown: list[str] = []
        usage = UsageInfo()
        for page_index in range(page_count):
            image = document.image
            try:
                if document.pdf_data is not None:
                    image = await asyncio.to_thread(
                        _render_pdf_page,
                        document.pdf_data,
                        page_index,
                    )
                assert image is not None
                page_result = await self._process_page(request, image)
                if isinstance(page_result, ErrorResponse):
                    message = page_result.error.message
                    logger.warning(
                        "OCR processing failed for page %d: %s",
                        page_index + 1,
                        message,
                    )
                    page_markdown.append(_page_failure_marker(page_index + 1, message))
                    continue
                markdown, page_usage = page_result
                page_markdown.append(markdown)
                usage.prompt_tokens += page_usage.prompt_tokens
                usage.completion_tokens = (usage.completion_tokens or 0) + (
                    page_usage.completion_tokens or 0
                )
                usage.total_tokens += page_usage.total_tokens
            except ValueError as exc:
                logger.warning(
                    "OCR processing failed for page %d: %s", page_index + 1, exc
                )
                page_markdown.append(_page_failure_marker(page_index + 1, str(exc)))
            except Exception as exc:
                logger.exception("OCR processing failed for page %d", page_index + 1)
                page_markdown.append(_page_failure_marker(page_index + 1, str(exc)))
            finally:
                if image is not None:
                    image.close()

        if page_count == 1:
            markdown = page_markdown[0]
        else:
            markdown = "\n\n---\n\n".join(
                content
                if content.startswith(f"<!-- Page {index} failed:")
                else f"<!-- Page {index} -->\n\n{content}"
                for index, content in enumerate(page_markdown, start=1)
            )
        response = self._make_response(request, markdown, usage=usage)
        if request.stream:
            return self._stream_response(response)
        return response

    async def _process_page(
        self, request: ChatCompletionRequest, image: Image.Image
    ) -> tuple[str, UsageInfo] | ErrorResponse:
        blocks = await self.layout.parse(image)
        try:
            return await self._process_blocks(request, blocks)
        finally:
            for block in blocks:
                block.image.close()

    async def _process_blocks(
        self, request: ChatCompletionRequest, blocks: list[OCRBlock]
    ) -> tuple[str, UsageInfo] | ErrorResponse:
        infer_blocks = [block for block in blocks if block.label not in _SKIP_LABELS]
        if not infer_blocks:
            return _assemble_markdown(blocks), UsageInfo()

        usage = UsageInfo()
        for block in infer_blocks:
            prompt, max_tokens, output_format = _ELEMENT_CONFIG.get(
                block.label, ("Text Recognition:", 8192, "text")
            )
            crop_request = ChatCompletionRequest(
                model=request.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": encode_image_url(block.image)},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=min(max_tokens, self.max_tokens),
                temperature=0.01,
                top_p=0.001,
                top_k=1,
                repetition_penalty=1.0,
                chat_template_kwargs={"enable_thinking": False},
                mm_processor_kwargs={
                    "downsample_mode": "4x",
                    "max_slice_nums": self.max_slice_nums,
                },
                priority=request.priority,
                stream=False,
            )
            crop_response = await self._infer_crop(crop_request)
            if isinstance(crop_response, ErrorResponse):
                return crop_response
            if not isinstance(crop_response, ChatCompletionResponse):
                return self.batch_serving.create_error_response(
                    "Unexpected streaming response from OCR crop inference",
                    err_type="InternalServerError",
                    status_code=500,
                )

            content = crop_response.choices[0].message.content
            raw_content = content if isinstance(content, str) else ""
            block.content = _postprocess_content(
                raw_content, output_format, block.label
            )
            usage.prompt_tokens += crop_response.usage.prompt_tokens
            usage.completion_tokens = (usage.completion_tokens or 0) + (
                crop_response.usage.completion_tokens or 0
            )
            usage.total_tokens += crop_response.usage.total_tokens
        return _assemble_markdown(blocks), usage

    async def _infer_crop(
        self, crop_request: ChatCompletionRequest
    ) -> ChatCompletionResponse | ErrorResponse | AsyncGenerator[str, None]:
        while True:
            response = await self.batch_serving.create_chat_completion(
                crop_request, None
            )
            max_tokens = crop_request.max_tokens or 0
            error = getattr(response, "error", None)
            if (
                error is not None
                and error.code == 400
                and "maximum context length" in error.message
                and max_tokens > 256
            ):
                crop_request.max_tokens = max(256, max_tokens // 2)
                continue
            return response

    def _make_response(
        self, request: ChatCompletionRequest, content: str, usage: Any | None = None
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=request.model or self.batch_serving.models.base_model_paths[0].name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=usage or UsageInfo(),
        )

    async def _stream_response(
        self, response: ChatCompletionResponse
    ) -> AsyncGenerator[str, None]:
        base = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
        }
        yield (
            "data: "
            + json.dumps(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        content = response.choices[0].message.content or ""
        for offset in range(0, len(content), 2048):
            yield (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content[offset : offset + 2048]},
                                "finish_reason": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"
