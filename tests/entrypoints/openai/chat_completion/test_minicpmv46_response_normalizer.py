# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.minicpmv46_response_normalizer import (
    EscapedNewlineNormalizer,
    MiniCPMV46ResponseNormalizer,
    normalize_response_text,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def test_normalize_response_text() -> None:
    text = (
        "first\\nsecond\\rthird\\r\\nfourth "
        "`code\\nvalue` "
        "```python\\nvalue = '\\\\n'\\n``` "
        "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] "
        "escaped \\\\n"
    )

    assert normalize_response_text(text) == (
        "first\nsecond\nthird\nfourth "
        "`code\\nvalue` "
        "```python\\nvalue = '\\\\n'\\n``` "
        "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] "
        "escaped \\\\n"
    )


def test_streaming_matches_complete_normalization() -> None:
    text = "first\\nsecond `code\\nvalue` $x\\ny$ ```text\\nvalue\\n``` last\\r\\nline"
    expected = normalize_response_text(text)

    for split in range(len(text) + 1):
        normalizer = EscapedNewlineNormalizer()
        actual = normalizer.feed(text[:split])
        actual += normalizer.feed(text[split:], final=True)
        assert actual == expected

    normalizer = EscapedNewlineNormalizer()
    actual = "".join(normalizer.feed(char) for char in text)
    actual += normalizer.feed("", final=True)
    assert actual == expected


def test_streaming_normalizes_reasoning_and_content() -> None:
    normalizer = MiniCPMV46ResponseNormalizer()

    first = normalizer.normalize_delta(
        DeltaMessage(reasoning="reason\\"),
        finished=False,
    )
    second = normalizer.normalize_delta(
        DeltaMessage(reasoning="nnext\\", content="answer\\"),
        finished=False,
    )
    final = normalizer.normalize_delta(
        DeltaMessage(content="nend"),
        finished=True,
    )

    assert first == DeltaMessage(reasoning="reason")
    assert second == DeltaMessage(reasoning="\nnext\\", content="answer")
    assert final == DeltaMessage(content="\nend")


def test_streaming_does_not_modify_tool_arguments() -> None:
    normalizer = MiniCPMV46ResponseNormalizer()
    arguments = '{"value":"first\\\\nsecond"}'
    delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=arguments),
            )
        ]
    )

    result = normalizer.normalize_delta(delta, finished=True)

    assert result is not None
    assert result.tool_calls[0].function is not None
    assert result.tool_calls[0].function.arguments == arguments
