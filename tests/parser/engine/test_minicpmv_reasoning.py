# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MiniCPM-V family reasoning parser."""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.minicpmv import MiniCPMVParser
from vllm.reasoning import ReasoningParserManager

_VOCAB = {
    "<think>": 10,
    "</think>": 11,
    "<|im_end|>": 12,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_VOCAB)


def test_reasoning_parser_is_registered():
    parser_cls = ReasoningParserManager.get_reasoning_parser("minicpmv")

    assert parser_cls.__name__ == "MiniCPMVParserReasoningAdapter"


class TestNonThinking:
    @pytest.fixture
    def parser(self, mock_tokenizer):
        return MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )

    def test_plain_content_is_unchanged(self, parser):
        reasoning, content = parser.extract_reasoning("The answer is 42.", None)

        assert reasoning is None
        assert content == "The answer is 42."

    def test_reserved_markers_are_removed(self, parser):
        output = (
            "<reserved_12>reasoning<reserved_13><|reserved_14|>tool call<|reserved_15|>"
        )

        reasoning, content = parser.extract_reasoning(output, None)

        assert reasoning is None
        assert content == "reasoningtool call"

    def test_unrelated_reserved_text_is_unchanged(self, parser):
        reasoning, content = parser.extract_reasoning(
            "<reserved_11>content<|reserved_16|>",
            None,
        )

        assert reasoning is None
        assert content == "<reserved_11>content<|reserved_16|>"

    def test_dangling_standard_think_end_is_removed(self, parser):
        output = "first part </think>\n\nfinal part"

        reasoning, content = parser.extract_reasoning(output, None)

        assert reasoning is None
        assert content == "first part \n\nfinal part"

    def test_truncated_standard_reasoning_does_not_leak(self, parser):
        reasoning, content = parser.extract_reasoning(
            "prefix<think>private reasoning",
            None,
        )

        assert reasoning is None
        assert content == "prefix"

    def test_streaming_split_standard_markers(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "<thi",
                "nk>",
                "private reasoning",
                "</thi",
                "nk>",
                "final answer",
            ],
        )

        assert reasoning == ""
        assert content == "final answer"

    def test_streaming_split_reserved_markers(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "<reser",
                "ved_12>",
                "reasoning",
                "<|reserved",
                "_13|>",
                "final answer",
            ],
        )

        assert reasoning == ""
        assert content == "reasoningfinal answer"


class TestThinking:
    def test_standard_think_tags_are_parsed(self, mock_tokenizer):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = parser.extract_reasoning(
            "<think>private reasoning</think>final answer",
            None,
        )

        assert reasoning == "private reasoning"
        assert content == "final answer"

    def test_reserved_markers_are_removed(self, mock_tokenizer):
        parser = MiniCPMVParser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = parser.extract_reasoning(
            "<think><reserved_12>private reasoning</think><|reserved_13|>final answer",
            None,
        )

        assert reasoning == "private reasoning"
        assert content == "final answer"

    def test_streaming_im_end_is_removed(self):
        tokenizer = make_mock_tokenizer(_VOCAB, special_tokens=[])
        parser = MiniCPMVParser(
            tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )

        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["Answer: 111", "<|im_end|>"],
        )

        assert reasoning == "Answer: 111"
        assert content == ""
