from __future__ import annotations

import json

import pytest

from ams.api import (
    GenerationCompleted,
    GenerationUsage,
    OpenAIStreamSession,
    ReasoningDelta,
    ResponseIdentity,
    TextDelta,
    ToolArgumentsDelta,
    ToolCallStart,
    normalize_chat_completions_request,
    normalize_responses_request,
    sse_data,
)
from ams.errors import AmsError, ErrorCode

IDENTITY = ResponseIdentity("resp_fixed", "msg_fixed", "rs_fixed", 1_234_567_890)


def _event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def test_responses_stream_reconstructs_exact_text_and_publishes_terminal_usage() -> None:
    request = normalize_responses_request(
        {
            "model": "ams-glm-4.7-flash",
            "input": "render this",
            "stream": True,
        }
    )
    session = OpenAIStreamSession(request, IDENTITY)
    events = list(session.start())
    for event in (
        TextDelta("Here is a flow:\n\n```mermaid\n"),
        TextDelta("flowchart TD\n  A --> B\n```\n"),
        GenerationCompleted(GenerationUsage(12, 9)),
    ):
        events.extend(session.push(event))

    assert (
        "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        == "Here is a flow:\n\n```mermaid\nflowchart TD\n  A --> B\n```\n"
    )
    assert _event_types(events) == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    terminal = events[-1]["response"]
    assert terminal == session.response_body()
    assert terminal["usage"] == {
        "input_tokens": 12,
        "output_tokens": 9,
        "total_tokens": 21,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert sse_data(events[0]).startswith(b'data: {"response":')
    assert sse_data("[DONE]") == b"data: [DONE]\n\n"


def test_tool_stream_maps_output_before_arguments_and_rejects_bad_backend_order() -> None:
    request = normalize_responses_request(
        {
            "model": "ams-glm-4.7-flash",
            "input": "read it",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": "auto",
            "stream": True,
        }
    )
    session = OpenAIStreamSession(request, IDENTITY)
    events = list(session.start())
    for event in (
        ReasoningDelta("I should inspect the file."),
        ToolCallStart(0, "call_1", "read_file"),
        ToolArgumentsDelta(0, '{"path":'),
        ToolArgumentsDelta(0, '"README.md"}'),
        GenerationCompleted(GenerationUsage(20, 8, reasoning_tokens=4), "tool_calls"),
    ):
        events.extend(session.push(event))

    types = _event_types(events)
    added = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
    )
    argument_deltas = [
        index
        for index, event in enumerate(events)
        if event["type"] == "response.function_call_arguments.delta"
    ]
    assert all(added < index for index in argument_deltas)
    assert types[-3:] == [
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    tool = events[-1]["response"]["output"][1]
    assert tool["call_id"] == "call_1"
    assert tool["name"] == "read_file"
    assert json.loads(tool["arguments"]) == {"path": "README.md"}

    invalid = OpenAIStreamSession(request, IDENTITY)
    with pytest.raises(AmsError) as error:
        invalid.push(ToolArgumentsDelta(0, "{}"))
    assert error.value.code is ErrorCode.INTERNAL_INVARIANT

    duplicate_arguments = OpenAIStreamSession(request, IDENTITY)
    duplicate_arguments.push(ToolCallStart(0, "call_2", "read_file"))
    duplicate_arguments.push(ToolArgumentsDelta(0, '{"path":"one","path":"two"}'))
    with pytest.raises(AmsError) as duplicate:
        duplicate_arguments.push(GenerationCompleted(GenerationUsage(20, 8), "tool_calls"))
    assert duplicate.value.code is ErrorCode.INTERNAL_INVARIANT


def test_chat_stream_and_nonstream_share_one_transactional_snapshot() -> None:
    request = normalize_chat_completions_request(
        {
            "model": "ams-glm-4.7-flash",
            "messages": [{"role": "user", "content": "answer"}],
            "stream": True,
        }
    )
    session = OpenAIStreamSession(request, IDENTITY)
    chunks = []
    for event in (
        ReasoningDelta("brief thought"),
        TextDelta("final "),
        TextDelta("answer"),
        GenerationCompleted(GenerationUsage(3, 4, reasoning_tokens=2)),
    ):
        chunks.extend(session.push(event))

    assert chunks[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "reasoning_content": "brief thought",
    }
    assert (
        "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        == "final answer"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    body = session.response_body()
    assert body["choices"][0]["message"]["content"] == "final answer"
    assert body["choices"][0]["message"]["reasoning_content"] == "brief thought"


def test_structured_output_failure_never_publishes_a_final_snapshot() -> None:
    request = normalize_responses_request(
        {
            "model": "ams-glm-4.7-flash",
            "input": "return json",
            "text": {"format": {"type": "json_object"}},
        }
    )
    session = OpenAIStreamSession(request, IDENTITY)
    session.push(TextDelta("not json"))
    with pytest.raises(AmsError) as error:
        session.push(GenerationCompleted(GenerationUsage(2, 2)))
    assert error.value.code is ErrorCode.INTERNAL_INVARIANT
    assert not session.accumulator.completed
