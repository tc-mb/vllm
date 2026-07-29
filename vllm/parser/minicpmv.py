# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-V family reasoning parser."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.qwen3 import Qwen3Parser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


@functools.cache
def minicpmv_no_thinking_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="minicpmv_no_thinking",
        initial_state=ParserState.CONTENT,
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_START"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_END"): Transition(ParserState.CONTENT),
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
        },
        strip_trailing_reasoning_whitespace=False,
    )


class MiniCPMVParser(Qwen3Parser):
    """Parse MiniCPM-V family output in thinking and non-thinking modes."""

    CONFIG_NAME = "minicpmv"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        if not chat_kwargs.get("enable_thinking", False):
            kwargs.setdefault(
                "parser_engine_config",
                minicpmv_no_thinking_config(),
            )
        super().__init__(tokenizer, tools, **kwargs)

    @property
    def reasoning_ended(self) -> bool:
        if not self.thinking_enabled:
            return False
        return super().reasoning_ended

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return False
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self.thinking_enabled:
            return super().extract_reasoning(model_output, request)
        _, content = ParserEngine.extract_reasoning(self, model_output, request)
        return None, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.thinking_enabled:
            return super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        delta = ParserEngine.extract_reasoning_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if delta is None:
            return None
        delta.reasoning = None
        if delta.content is None and not delta.tool_calls:
            return None
        return delta

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        if self.thinking_enabled:
            return super().get_streaming_fallback_content(text, request)
        delta = ParserEngine.finish_streaming(self)
        return delta.content if delta is not None else None
