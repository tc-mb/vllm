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
from vllm.entrypoints.openai.chat_completion.otsl import convert_otsl_to_html
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
_WORD_RE = re.compile(r"\S+")
_EMPTY_BRACKET_MATH_RE = re.compile(r"\\\[\s*\\\]|\\\(\s*\\\)")
_DISPLAY_SPAN_RE = re.compile(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.DOTALL)
_DISPLAY_DOLLAR_SPLIT_RE = re.compile(r"(\$\$)")
_DISPLAY_DOLLAR_RE = re.compile(r"(?<!\\)\$\$")
_INLINE_DOLLAR_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)")
_TRAILING_BACKSLASH_RE = re.compile(r"(?<!\\)\\$")
_BEGIN_RE = re.compile(r"\\begin\{")
_END_RE = re.compile(r"\\end\{")
_INLINE_MATH_RE = re.compile(
    r"(?<!\\)\$(?!\$)(?:[^$\n]|\\.)+?(?<!\\)\$(?!\$)|\\\([^\n]*?\\\)"
)
_LATEX_SPACING_RE = re.compile(
    r"\\(?:quad|qquad|enspace|thinspace|medspace|thickspace|"
    r"negthinspace|negmedspace|negthickspace)(?![A-Za-z])"
)
_LATEX_SHORT_SPACE_RE = re.compile(r"\\[,;:!]")
_LATEX_BACKSLASH_SPACE_RE = re.compile(r"\\ ")
_LATEX_HVSPACE_RE = re.compile(r"\\(?:hspace|vspace)\s*\{[^{}]*\}")
_LATEX_STRIPPED_CHARS_RE = re.compile(r"[\s{}]")
_PDF_SCALE = 2.0

# Labels whose crops hold ordinary prose: line layout inside them is an
# artefact of the crop, so the text is re-flowed into paragraphs.
_TEXT_BLOCK_LABELS = {
    "text",
    "content",
    "abstract",
    "reference",
    "reference_content",
    "vertical_text",
    "vision_footnote",
    "algorithm",
    "header",
    "footer",
    "footnote",
    "aside_text",
}

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
    inline_formula_mixed: bool = False
    raw_content: str = ""
    output_token_count: int = 0
    status: str = "ok"


@dataclass
class OCRDocument:
    image: Image.Image | None = None
    pdf_data: bytes | None = None


@dataclass
class OCRPageLayout:
    """Layout-detection result for one page, plus the OCR blocks."""

    blocks: list["OCRBlock"]
    raw_boxes: list[dict]      # filtered pre-merge boxes (cls_id/score intact)
    image_width: int
    image_height: int


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

    def _parse(self, image: Image.Image) -> OCRPageLayout:
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

        # Recover cls_id / score / polygon by coordinate key before merge_blocks
        # strips them (same technique as pip's ppdoclayout3_paddlex adapter).
        meta_by_coord: dict[tuple, tuple] = {
            tuple(float(v) for v in b["coordinate"]): (
                int(b["cls_id"]),
                float(b["score"]),
                b.get("polygon_points"),
            )
            for b in boxes
        }
        # Keep a serialisable copy of filtered boxes for layout_det_res output.
        filtered_raw_boxes = [
            {
                "cls_id": int(b["cls_id"]),
                "label": b["label"],
                "score": round(float(b["score"]), 6),
                "coordinate": [round(float(v), 2) for v in b["coordinate"]],
                "polygon_points": _polygon_to_list(b.get("polygon_points")),
            }
            for b in boxes
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
            coord_key = tuple(float(v) for v in block["box"])
            cls_id, score, polygon = meta_by_coord.get(coord_key, (-1, 0.0, None))
            blocks.append(
                OCRBlock(
                    label=_LABEL_MAP.get(raw_label, raw_label),
                    order=idx,
                    bbox=(x1, y1, x2, y2),
                    image=Image.fromarray(crop),
                )
            )
            # Stash layout metadata directly on the block so it travels through
            # _process_blocks without needing a separate lookup table.
            blocks[-1]._cls_id = cls_id
            blocks[-1]._score = score
            blocks[-1]._polygon = _polygon_to_list(polygon)
        return OCRPageLayout(
            blocks=blocks,
            raw_boxes=filtered_raw_boxes,
            image_width=image.width,
            image_height=image.height,
        )

    async def parse(self, image: Image.Image) -> OCRPageLayout:
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


def _truncate_repetitive_suffix(text: str) -> str:
    """Cut runaway generation loops: three consecutive identical word windows.

    Concurrent batching occasionally flips a long table/text crop into a
    repeat loop that runs to max_tokens; this trims the looped suffix so the
    assembled markdown stays clean even when decoding already paid the cost.
    """
    words = list(_WORD_RE.finditer(text))
    if len(words) < 18:
        return text

    normalized = [match.group(0).lower() for match in words]
    earliest_cut = len(text)
    max_window = min(64, len(words) // 3)
    for window in range(6, max_window + 1):
        limit = len(words) - (window * 3) + 1
        for start in range(limit):
            first = normalized[start : start + window]
            if (
                first == normalized[start + window : start + (window * 2)]
                and first == normalized[start + (window * 2) : start + (window * 3)]
            ):
                earliest_cut = min(earliest_cut, words[start].start())
                break

    if earliest_cut < len(text):
        return text[:earliest_cut].rstrip()
    return text


def _truncate_repetitive_content(content: str, min_count: int) -> str:
    if not content or len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content

    # A fixed-length unit looping at the tail over more than half the string.
    if "\n" not in stripped and len(stripped) > 100:
        for unit_len in range(8, len(stripped) // 5 + 1):
            unit = stripped[-unit_len:]
            count = 0
            pos = len(stripped) - unit_len
            while pos >= 0 and stripped[pos : pos + unit_len] == unit:
                count += 1
                pos -= unit_len
            if count >= 5 and len(unit) * count > len(stripped) * 0.5:
                return stripped[: len(stripped) - (count * unit_len)]

    # The whole string is one short unit repeated over and over.
    if "\n" not in stripped and len(stripped) > 10:
        for unit_len in range(1, len(stripped) // 2 + 1):
            repeats = len(stripped) // unit_len
            unit = stripped[:unit_len]
            if unit * repeats == stripped[: unit_len * repeats]:
                if repeats >= 10:
                    return unit
                break

    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if len(lines) >= 10:
        most_common, count = Counter(lines).most_common(1)[0]
        if count >= 10 and count / len(lines) >= 0.8:
            return most_common
    return content


def _strip_spacing_tokens(content: str) -> str:
    content = _LATEX_SPACING_RE.sub("", content)
    content = _LATEX_SHORT_SPACE_RE.sub("", content)
    content = _LATEX_BACKSLASH_SPACE_RE.sub("", content)
    content = _LATEX_HVSPACE_RE.sub("", content)
    return _LATEX_STRIPPED_CHARS_RE.sub("", content.replace("~", ""))


def _repair_display_dollars(markdown: str) -> str:
    """Drop unpaired and empty ``$$`` delimiters left by a truncated crop."""
    tokens = _DISPLAY_DOLLAR_SPLIT_RE.split(markdown)
    output: list[str] = []
    in_display = False
    for index, token in enumerate(tokens):
        if token != "$$":
            output.append(token)
            continue
        next_segment = tokens[index + 1] if index + 1 < len(tokens) else ""
        next_is_delimiter = index + 2 < len(tokens) and tokens[index + 2] == "$$"
        has_later_delimiter = any(part == "$$" for part in tokens[index + 2 :])
        if not in_display:
            if (
                not next_segment.strip() and next_is_delimiter
            ) or not has_later_delimiter:
                continue
            output.append(token)
            in_display = True
        else:
            output.append(token)
            in_display = False
    if in_display:
        output.pop()
    return "".join(output)


def _repair_display_formula(content: str) -> str:
    """Repair malformed display math without canonicalizing formula content."""
    content = _repair_display_dollars(content)
    content = _EMPTY_BRACKET_MATH_RE.sub(" ", content)

    def clean_span(match: re.Match[str]) -> str:
        whole = match.group(0)
        is_bracket = whole.startswith(r"\[")
        body = match.group(1) if is_bracket else match.group(2)
        stripped = body.rstrip()
        trailing_whitespace = body[len(stripped) :]
        if _TRAILING_BACKSLASH_RE.search(stripped):
            body = stripped[:-1] + trailing_whitespace
        if body.strip() and not _strip_spacing_tokens(body):
            return " "
        return (r"\[" + body + r"\]") if is_bracket else f"$${body}$$"

    return _DISPLAY_SPAN_RE.sub(clean_span, content)


def _has_mixed_inline_formula_content(content: str) -> bool:
    """Whether an inline-formula crop also contains ordinary text."""
    stripped = (content or "").strip()
    if not stripped:
        return False
    if _INLINE_MATH_RE.fullmatch(stripped):
        return False
    without_formulas = _INLINE_MATH_RE.sub("", stripped)
    return bool(without_formulas.strip())


def _is_pipe_table_row(line: str) -> bool:
    stripped = line.strip()
    return (stripped.count("|") >= 2 and stripped.startswith("|")) or (
        stripped.count("|") >= 2 and "---" in stripped
    )


def _split_text_block(content: str, *, protect_inline_math: bool = False) -> str:
    """Re-flow prose lines while preserving Markdown and math structures.

    A crop's line breaks come from the page layout, not from the text, so
    every plain line becomes its own paragraph. Fenced code, HTML and pipe
    tables, display math and LaTeX environments are passed through untouched.
    """
    lines = content.split("\n")
    output: list[str] = []
    in_code = in_dollar = in_bracket = in_html_table = False
    in_inline_dollar = in_inline_paren = False
    env_depth = 0

    def protected() -> bool:
        return (
            in_code
            or in_dollar
            or in_bracket
            or in_html_table
            or in_inline_dollar
            or in_inline_paren
            or env_depth > 0
        )

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            output.append(line)
            in_code = not in_code
            continue
        if in_code:
            output.append(line)
            continue

        if "<table" in stripped.lower():
            in_html_table = True
        if in_html_table:
            output.append(line)
            if "</table>" in stripped.lower():
                in_html_table = False
            continue

        if _is_pipe_table_row(line):
            output.append(line)
            next_is_pipe_row = index + 1 < len(lines) and _is_pipe_table_row(
                lines[index + 1]
            )
            if not next_is_pipe_row and output[-1].strip():
                output.append("")
            continue

        display_dollar_count = len(_DISPLAY_DOLLAR_RE.findall(line))
        bracket_opens = r"\[" in line and r"\]" not in line.split(r"\[", 1)[1]
        inline_dollar_count = len(_INLINE_DOLLAR_RE.findall(line))
        inline_paren_opens = r"\(" in line and r"\)" not in line.split(r"\(", 1)[1]

        if protected():
            output.append(line)
            if in_dollar and display_dollar_count % 2:
                in_dollar = False
            if in_bracket and r"\]" in line:
                in_bracket = False
            if in_inline_dollar and inline_dollar_count % 2:
                in_inline_dollar = False
            if in_inline_paren and r"\)" in line:
                in_inline_paren = False
            env_depth = max(
                0, env_depth + len(_BEGIN_RE.findall(line)) - len(_END_RE.findall(line))
            )
            continue

        net_env = len(_BEGIN_RE.findall(line)) - len(_END_RE.findall(line))
        if display_dollar_count % 2 or bracket_opens or net_env > 0:
            output.append(line)
            in_dollar = display_dollar_count % 2 == 1
            in_bracket = bracket_opens
            env_depth = max(0, env_depth + net_env)
            continue
        if protect_inline_math and (inline_dollar_count % 2 or inline_paren_opens):
            output.append(line)
            in_inline_dollar = inline_dollar_count % 2 == 1
            in_inline_paren = inline_paren_opens
            continue

        if stripped:
            if output and output[-1].strip():
                output.append("")
            output.extend((stripped, ""))

    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip("\n")


def _postprocess_content(
    content: str,
    output_format: str,
    label: str,
    inline_formula_mixed: bool = False,
) -> str:
    content = _truncate_repetitive_content(
        content.strip(), 5000 if label == "table" else 50
    )
    if output_format == "latex":
        if label == "display_formula":
            content = _repair_display_formula(content)
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if content.startswith(left) and content.endswith(right):
                content = content[len(left) : -len(right)].strip()
        if label == "display_formula":
            content = _DISPLAY_BRACKET_SEP_RE.sub("$$   $$", content)
            content = content.replace(r"\[", "").replace(r"\]", "")
    elif output_format == "html":
        if content.startswith("```html"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if label == "table":
            content += "</table>" * max(
                0, content.count("<table") - content.count("</table>")
            )
    content = _truncate_repetitive_suffix(content.strip())
    if label != "display_formula":
        content = _INLINE_LATEX_RE.sub(
            lambda match: f"${match.group(1).strip()}$", content
        )
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    if label == "table":
        # OTSL output converts to HTML; anything else falls through unchanged.
        table_html = convert_otsl_to_html(content)
        if table_html:
            return table_html
    if label in _TEXT_BLOCK_LABELS:
        content = _split_text_block(content)
    elif label == "inline_formula" and inline_formula_mixed:
        content = _split_text_block(content, protect_inline_math=True)
    return content


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
            parts.append(content if block.inline_formula_mixed else f"${content}$")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _polygon_to_list(polygon) -> list[list[float]]:
    """Convert numpy polygon array or None to a plain list of [x, y] pairs."""
    if polygon is None:
        return []
    try:
        import numpy as np  # already imported at module level; guarded for safety
        arr = np.asarray(polygon, dtype=float).reshape(-1, 2)
        return [[round(float(r[0]), 2), round(float(r[1]), 2)] for r in arr]
    except Exception:
        return []


def _build_page_layout_json(
    page_layout: OCRPageLayout,
    blocks: list[OCRBlock],
    page_index: int,
) -> dict:
    """Build a PaddleX-compatible layout dict for one page.

    The structure mirrors ``layoutParsingResults[n].prunedResult`` from the
    paddle_ocr_parsing_demo.json reference, keeping ``parsing_res_list`` and
    ``layout_det_res`` as the two main containers.
    """
    parsing_res_list: list[dict] = []
    for block in sorted(blocks, key=lambda b: b.order):
        x1, y1, x2, y2 = block.bbox
        poly = getattr(block, "_polygon", None) or [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ]
        parsing_res_list.append(
            {
                "block_label": block.label,
                "block_content": block.content,
                "block_bbox": list(block.bbox),
                "block_id": block.order,
                "block_order": (
                    None if block.label in _IGNORE_LABELS else block.order
                ),
                "group_id": block.order,
                "block_polygon_points": poly,
                # vllm-specific extras (not in the paddle reference schema)
                "block_status": block.status,
                "raw_content": block.raw_content,
                "output_token_count": block.output_token_count,
                "inline_formula_mixed": block.inline_formula_mixed,
            }
        )

    boxes: list[dict] = []
    for order, rb in enumerate(page_layout.raw_boxes):
        boxes.append(
            {
                "cls_id": rb["cls_id"],
                "label": rb["label"],
                "score": rb["score"],
                "coordinate": rb["coordinate"],
                "order": order,
                "polygon_points": rb["polygon_points"],
            }
        )

    return {
        "page_index": page_index,
        "width": page_layout.image_width,
        "height": page_layout.image_height,
        "parsing_res_list": parsing_res_list,
        "layout_det_res": {"boxes": boxes},
    }


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
        crop_concurrency: int = 16,
    ):
        if max_pdf_pages < 0:
            raise ValueError("ocr_max_pdf_pages must not be negative")
        if crop_concurrency < 1:
            raise ValueError("ocr_crop_concurrency must be >= 1")
        self.batch_serving = batch_serving
        self.layout = PPDocLayoutService(layout_model, layout_device, max_crops)
        self.max_tokens = max_tokens
        self.max_slice_nums = max_slice_nums
        self.max_pdf_pages = max_pdf_pages
        self.crop_concurrency = crop_concurrency
        self._crop_sema = asyncio.Semaphore(crop_concurrency)
        logger.info(
            "OCR pipeline crop_concurrency=%d layout_device=%s",
            crop_concurrency,
            layout_device,
        )

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
        page_layouts: list[OCRPageLayout] = []   # parallel to page_markdown
        usage = UsageInfo()
        pending_layout: asyncio.Task[OCRPageLayout] | None = None
        pending_image: Image.Image | None = None
        for page_index in range(page_count):
            image: Image.Image | None = None
            cur_page_layout: OCRPageLayout | None = None
            blocks: list[OCRBlock] = []
            try:
                if pending_layout is not None:
                    image = pending_image
                    pending_image = None
                    cur_page_layout = await pending_layout
                    pending_layout = None
                else:
                    if document.pdf_data is not None:
                        image = await asyncio.to_thread(
                            _render_pdf_page,
                            document.pdf_data,
                            page_index,
                        )
                    else:
                        image = document.image
                    assert image is not None
                    cur_page_layout = await self.layout.parse(image)
                blocks = cur_page_layout.blocks
                if document.pdf_data is not None and page_index + 1 < page_count:
                    pending_image = await asyncio.to_thread(
                        _render_pdf_page,
                        document.pdf_data,
                        page_index + 1,
                    )
                    pending_layout = asyncio.create_task(
                        self.layout.parse(pending_image)
                    )
                page_result = await self._process_blocks(request, blocks)
                if isinstance(page_result, ErrorResponse):
                    message = page_result.error.message
                    logger.warning(
                        "OCR processing failed for page %d: %s",
                        page_index + 1,
                        message,
                    )
                    page_markdown.append(_page_failure_marker(page_index + 1, message))
                    page_layouts.append(
                        cur_page_layout
                        if cur_page_layout is not None
                        else OCRPageLayout([], [], 0, 0)
                    )
                    continue
                markdown, page_usage = page_result
                page_markdown.append(markdown)
                page_layouts.append(
                    cur_page_layout
                    if cur_page_layout is not None
                    else OCRPageLayout([], [], 0, 0)
                )
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
                page_layouts.append(OCRPageLayout([], [], 0, 0))
            except Exception as exc:
                logger.exception("OCR processing failed for page %d", page_index + 1)
                page_markdown.append(_page_failure_marker(page_index + 1, str(exc)))
                page_layouts.append(OCRPageLayout([], [], 0, 0))
            finally:
                for block in blocks:
                    block.image.close()
                if image is not None:
                    image.close()
        if pending_layout is not None:
            pending_layout.cancel()
        if pending_image is not None:
            pending_image.close()

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
        if request.ocr_return_layout:
            response.ocr_layout = [
                _build_page_layout_json(pl, pl.blocks, idx + 1)
                for idx, pl in enumerate(page_layouts)
            ]
        if request.stream:
            return self._stream_response(response)
        return response

    async def _process_page(
        self, request: ChatCompletionRequest, image: Image.Image
    ) -> tuple[str, UsageInfo] | ErrorResponse:
        page_layout = await self.layout.parse(image)
        try:
            return await self._process_blocks(request, page_layout.blocks)
        finally:
            for block in page_layout.blocks:
                block.image.close()

    async def _process_blocks(
        self, request: ChatCompletionRequest, blocks: list[OCRBlock]
    ) -> tuple[str, UsageInfo] | ErrorResponse:
        infer_blocks = [block for block in blocks if block.label not in _SKIP_LABELS]
        if not infer_blocks:
            return _assemble_markdown(blocks), UsageInfo()

        ordered = sorted(
            infer_blocks,
            key=lambda block: -(block.image.size[0] * block.image.size[1]),
        )

        async def _one(block: OCRBlock):
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
            async with self._crop_sema:
                crop_response = await self._infer_crop(crop_request)
            return block, output_format, crop_response

        results = await asyncio.gather(*[_one(block) for block in ordered])
        usage = UsageInfo()
        for block, output_format, crop_response in results:
            if not isinstance(crop_response, ChatCompletionResponse):
                # Unexpected streaming response is a server-side bug; fail page.
                if not isinstance(crop_response, ErrorResponse):
                    return self.batch_serving.create_error_response(
                        "Unexpected streaming response from OCR crop inference",
                        err_type="InternalServerError",
                        status_code=500,
                    )
                # Per-block error: degrade gracefully instead of failing page.
                logger.warning(
                    "OCR crop inference failed for block (label=%s order=%d): %s",
                    block.label,
                    block.order,
                    crop_response.error.message,
                )
                block.status = "error"
                block.raw_content = ""
                block.content = ""
                block.output_token_count = 0
                continue
            content = crop_response.choices[0].message.content
            raw_content = content if isinstance(content, str) else ""
            block.inline_formula_mixed = (
                block.label == "inline_formula"
                and _has_mixed_inline_formula_content(raw_content)
            )
            block.raw_content = raw_content
            block.output_token_count = crop_response.usage.completion_tokens or 0
            block.status = "ok"
            block.content = _postprocess_content(
                raw_content, output_format, block.label, block.inline_formula_mixed
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
