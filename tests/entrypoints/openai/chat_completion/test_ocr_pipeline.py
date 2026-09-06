# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image

from vllm.entrypoints.openai.chat_completion.ocr_pipeline import (
    OCRBlock,
    OCRPipelineServing,
    PPDocLayoutService,
    _assemble_markdown,
    _extract_document,
    _postprocess_content,
    _to_paddle_device,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    UsageInfo,
)
from vllm.multimodal.utils import encode_image_url


def make_request(*, stream: bool = False) -> ChatCompletionRequest:
    image = Image.new("RGB", (16, 12), "white")
    return ChatCompletionRequest(
        model="ocr-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image)},
                    },
                    {"type": "text", "text": "OCR this document"},
                ],
            }
        ],
        stream=stream,
    )


def make_pdf_request(pdf_data: bytes = b"%PDF-test") -> ChatCompletionRequest:
    encoded = base64.b64encode(pdf_data).decode()
    return ChatCompletionRequest(
        model="ocr-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_data": (f"data:application/pdf;base64,{encoded}"),
                        },
                    },
                    {"type": "text", "text": "OCR this document"},
                ],
            }
        ],
    )


def make_pipeline(batch_serving: Mock) -> OCRPipelineServing:
    return OCRPipelineServing(
        batch_serving=batch_serving,
        layout_model="/layout",
        layout_device="cpu",
        max_crops=8,
        max_tokens=8192,
        max_slice_nums=9,
        max_pdf_pages=8,
    )


def test_extract_data_image():
    document = _extract_document(make_request())

    assert document is not None
    assert document.image is not None
    assert document.image.size == (16, 12)


def test_extract_pdf_file():
    document = _extract_document(make_pdf_request())

    assert document is not None
    assert document.pdf_data == b"%PDF-test"


def test_postprocess_and_assemble_markdown():
    blocks = [
        OCRBlock("doc_title", 0, (0, 0, 1, 1), Image.new("RGB", (1, 1))),
        OCRBlock("display_formula", 1, (0, 0, 1, 1), Image.new("RGB", (1, 1))),
        OCRBlock("table", 2, (0, 0, 1, 1), Image.new("RGB", (1, 1))),
    ]
    blocks[0].content = "Title"
    blocks[1].content = _postprocess_content(r"\[x+y\]", "latex", "display_formula")
    blocks[2].content = _postprocess_content("<table><tr></tr>", "html", "table")

    assert _assemble_markdown(blocks) == (
        "# Title\n\n$$\nx+y\n$$\n\n<table><tr></tr></table>"
    )


def test_pipeline_batches_crops_and_returns_markdown():
    asyncio.run(_test_pipeline_batches_crops_and_returns_markdown())


async def _test_pipeline_batches_crops_and_returns_markdown():
    batch_serving = Mock()
    batch_serving.create_chat_completion = AsyncMock(
        side_effect=[
            ChatCompletionResponse(
                model="ocr-model",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Document title"),
                    )
                ],
                usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
            ChatCompletionResponse(
                model="ocr-model",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Body text"),
                    )
                ],
                usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]
    )
    pipeline = make_pipeline(batch_serving)
    pipeline.layout.parse = AsyncMock(
        return_value=[
            OCRBlock("doc_title", 0, (0, 0, 8, 4), Image.new("RGB", (8, 4), "white")),
            OCRBlock("text", 1, (0, 4, 8, 8), Image.new("RGB", (8, 4), "white")),
        ]
    )

    response = await pipeline.create_chat_completion(make_request(), Mock())

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == "# Document title\n\nBody text"
    assert batch_serving.create_chat_completion.call_count == 2
    crop_request = batch_serving.create_chat_completion.call_args_list[0].args[0]
    second_crop_request = batch_serving.create_chat_completion.call_args_list[1].args[0]
    assert crop_request.max_tokens == 512
    assert second_crop_request.max_tokens == 4096
    assert crop_request.repetition_penalty == 1.0
    assert crop_request.mm_processor_kwargs == {
        "downsample_mode": "4x",
        "max_slice_nums": 9,
    }


def test_pipeline_streams_openai_chunks():
    asyncio.run(_test_pipeline_streams_openai_chunks())


async def _test_pipeline_streams_openai_chunks():
    batch_serving = Mock()
    pipeline = make_pipeline(batch_serving)
    pipeline.layout.parse = AsyncMock(return_value=[])

    stream = await pipeline.create_chat_completion(make_request(stream=True), Mock())

    assert not isinstance(stream, ChatCompletionResponse)
    chunks = [chunk async for chunk in stream]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_pipeline_parses_pdf_pages():
    asyncio.run(_test_pipeline_parses_pdf_pages())


async def _test_pipeline_parses_pdf_pages():
    batch_serving = Mock()
    pipeline = make_pipeline(batch_serving)
    pipeline.layout.parse = AsyncMock(
        side_effect=[
            [OCRBlock("text", 0, (0, 0, 8, 4), Image.new("RGB", (8, 4), "white"))],
            [OCRBlock("text", 0, (0, 0, 8, 4), Image.new("RGB", (8, 4), "white"))],
        ]
    )
    batch_serving.create_chat_completion = AsyncMock(
        side_effect=[
            ChatCompletionResponse(
                model="ocr-model",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="First page"),
                    )
                ],
                usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
            ChatCompletionResponse(
                model="ocr-model",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Second page"),
                    )
                ],
                usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]
    )

    with (
        patch(
            "vllm.entrypoints.openai.chat_completion.ocr_pipeline._get_pdf_page_count",
            return_value=2,
        ),
        patch(
            "vllm.entrypoints.openai.chat_completion.ocr_pipeline._render_pdf_page",
            side_effect=[
                Image.new("RGB", (16, 12), "white"),
                Image.new("RGB", (16, 12), "white"),
            ],
        ),
    ):
        response = await pipeline.create_chat_completion(make_pdf_request(), Mock())

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == (
        "<!-- Page 1 -->\n\nFirst page\n\n---\n\n<!-- Page 2 -->\n\nSecond page"
    )
    assert response.usage.total_tokens == 14


def test_crop_context_overflow_halves_max_tokens():
    asyncio.run(_test_crop_context_overflow_halves_max_tokens())


async def _test_crop_context_overflow_halves_max_tokens():
    batch_serving = Mock()
    attempted_max_tokens = []
    success = ChatCompletionResponse(
        model="ocr-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="table"),
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    async def create_chat_completion(request, _):
        attempted_max_tokens.append(request.max_tokens)
        if len(attempted_max_tokens) == 1:
            return ErrorResponse(
                error=ErrorInfo(
                    message="This model's maximum context length is 8192 tokens",
                    type="BadRequestError",
                    code=400,
                )
            )
        return success

    batch_serving.create_chat_completion = create_chat_completion
    pipeline = make_pipeline(batch_serving)
    request = ChatCompletionRequest(
        model="ocr-model",
        messages=[{"role": "user", "content": "Table Recognition:"}],
        max_tokens=8192,
    )

    response = await pipeline._infer_crop(request)

    assert response is success
    assert attempted_max_tokens == [8192, 4096]


def test_pipeline_keeps_successful_pages_after_page_failure():
    asyncio.run(_test_pipeline_keeps_successful_pages_after_page_failure())


async def _test_pipeline_keeps_successful_pages_after_page_failure():
    batch_serving = Mock()
    batch_serving.create_chat_completion = AsyncMock(
        side_effect=[
            ErrorResponse(
                error=ErrorInfo(
                    message="page inference failed",
                    type="InternalServerError",
                    code=500,
                )
            ),
            ChatCompletionResponse(
                model="ocr-model",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Second page"),
                    )
                ],
                usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]
    )
    pipeline = make_pipeline(batch_serving)
    pipeline.layout.parse = AsyncMock(
        side_effect=[
            [OCRBlock("text", 0, (0, 0, 8, 4), Image.new("RGB", (8, 4), "white"))],
            [OCRBlock("text", 0, (0, 0, 8, 4), Image.new("RGB", (8, 4), "white"))],
        ]
    )

    with (
        patch(
            "vllm.entrypoints.openai.chat_completion.ocr_pipeline._get_pdf_page_count",
            return_value=2,
        ),
        patch(
            "vllm.entrypoints.openai.chat_completion.ocr_pipeline._render_pdf_page",
            side_effect=[
                Image.new("RGB", (16, 12), "white"),
                Image.new("RGB", (16, 12), "white"),
            ],
        ),
    ):
        response = await pipeline.create_chat_completion(make_pdf_request(), Mock())

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == (
        "<!-- Page 1 failed: page inference failed -->\n\n---\n\n"
        "<!-- Page 2 -->\n\nSecond page"
    )
    assert response.usage.total_tokens == 7


def test_to_paddle_device():
    assert _to_paddle_device("cpu") == "cpu"
    assert _to_paddle_device("CUDA") == "gpu:0"
    assert _to_paddle_device("cuda:1") == "gpu:1"
    assert _to_paddle_device("gpu:0") == "gpu:0"


def test_layout_parse_uses_paddlex_filter_crop_merge():
    service = PPDocLayoutService("/layout", "cpu", max_crops=0)
    crop = np.zeros((8, 6, 3), dtype=np.uint8)
    service._model = Mock()
    service._model.predict.return_value = [
        {
            "boxes": [
                {
                    "cls_id": 6,
                    "label": "doc_title",
                    "score": 0.9,
                    "coordinate": [1.2, 2.4, 7.6, 9.1],
                    "polygon_points": [[1, 2], [8, 2], [8, 9], [1, 9]],
                }
            ]
        }
    ]
    service._filter_overlap_boxes = Mock(side_effect=lambda data, _mode: data)
    service._crop_by_boxes = Mock(
        return_value=[{"label": "doc_title", "box": [1, 2, 8, 9], "img": crop}]
    )
    service._merge_blocks = Mock(side_effect=lambda cropped, **_kwargs: cropped)

    image = Image.new("RGB", (16, 12), "white")
    blocks = service._parse(image)

    assert [block.label for block in blocks] == ["doc_title"]
    assert blocks[0].bbox == (1, 2, 8, 9)
    assert blocks[0].image.size == (6, 8)
    predict_kwargs = service._model.predict.call_args.kwargs
    assert predict_kwargs["filter_overlap_boxes"] is False
    assert predict_kwargs["layout_shape_mode"] == "auto"
    np.testing.assert_array_equal(
        service._model.predict.call_args.args[0], np.asarray(image)
    )
    service._filter_overlap_boxes.assert_called_once()
    service._crop_by_boxes.assert_called_once()
    service._merge_blocks.assert_called_once()
    assert service._merge_blocks.call_args.kwargs["non_merge_labels"] == [
        "image",
        "header_image",
        "footer_image",
        "chart",
        "seal",
        "table",
    ]


def test_layout_parse_drops_folded_crops_and_maps_formula():
    service = PPDocLayoutService("/layout", "cpu", max_crops=0)
    service._model = Mock()
    service._model.predict.return_value = [{"boxes": []}]
    service._filter_overlap_boxes = Mock()
    service._crop_by_boxes = Mock()
    service._merge_blocks = Mock(
        return_value=[
            {
                "label": "formula",
                "box": [0, 0, 4, 4],
                "img": np.zeros((4, 4, 3), dtype=np.uint8),
            },
            {"label": "text", "box": [0, 4, 4, 8], "img": None},
        ]
    )

    blocks = service._parse(Image.new("RGB", (8, 8), "white"))

    assert len(blocks) == 1
    assert blocks[0].label == "display_formula"
    assert blocks[0].order == 0
    service._filter_overlap_boxes.assert_not_called()
    service._crop_by_boxes.assert_not_called()
