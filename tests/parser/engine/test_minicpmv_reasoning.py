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
