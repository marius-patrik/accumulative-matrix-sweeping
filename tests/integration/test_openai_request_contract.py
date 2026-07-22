from __future__ import annotations

import pytest

from ams.api import (
    FunctionCallItem,
    FunctionOutputItem,
    MessageItem,
    OpenAIEndpoint,
    OpenAIProtocolError,
    normalize_chat_completions_request,
    normalize_responses_request,
    parse_openai_json,
)
from ams.errors import ErrorCode

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def test_froq_responses_and_chat_requests_normalize_to_one_model_contract() -> None:
    responses = normalize_responses_request(
        {
            "model": "ams-glm-4.7-flash",
            "input": [
                {"type": "message", "role": "system", "content": "You are careful."},
                {"type": "message", "role": "user", "content": "Read the file."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "contents",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read one file",
                    "parameters": SCHEMA,
                }
            ],
            "tool_choice": "auto",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": SCHEMA,
                    "strict": True,
                }
            },
            "reasoning": {"effort": "high", "summary": "concise"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "stream": True,
            "max_output_tokens": 500,
            "temperature": 0.5,
        }
    )
    chat = normalize_chat_completions_request(
        {
            "model": "ams-glm-4.7-flash",
            "messages": [
                {"role": "system", "content": "You are careful."},
                {"role": "user", "content": "Read the file."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read one file",
                        "parameters": SCHEMA,
                    },
                }
            ],
            "tool_choice": "auto",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": SCHEMA,
                    "strict": True,
                },
            },
            "reasoning_effort": "high",
            "stream": True,
            "max_tokens": 500,
            "temperature": 0.5,
        }
    )

    assert responses.endpoint is OpenAIEndpoint.RESPONSES
    assert chat.endpoint is OpenAIEndpoint.CHAT_COMPLETIONS
    assert responses.input_items == chat.input_items
    assert [type(item) for item in responses.input_items] == [
        MessageItem,
        MessageItem,
        FunctionCallItem,
        FunctionOutputItem,
    ]
    assert responses.tools == chat.tools
    assert responses.tool_choice == chat.tool_choice
    assert responses.structured_output == chat.structured_output
    assert responses.max_output_tokens == chat.max_output_tokens == 500
    assert responses.temperature == chat.temperature == 0.5
    assert responses.reasoning_effort == chat.reasoning_effort == "high"
    assert responses.reasoning_summary == "concise"


def test_request_boundary_rejects_duplicate_keys_and_provider_field_drift() -> None:
    with pytest.raises(OpenAIProtocolError) as duplicate:
        parse_openai_json(b'{"model":"one","model":"two"}')
    assert duplicate.value.api_code == "invalid_json"

    with pytest.raises(OpenAIProtocolError) as drift:
        normalize_responses_request(
            {
                "model": "ams-glm-4.7-flash",
                "input": "hello",
                "future_provider_knob": True,
            }
        )
    assert drift.value.param == "future_provider_knob"
    assert drift.value.code is ErrorCode.UNSUPPORTED_OP


def test_tool_boundary_rejects_hosted_tools_and_undeclared_required_function() -> None:
    with pytest.raises(OpenAIProtocolError) as hosted:
        normalize_responses_request(
            {
                "model": "ams-glm-4.7-flash",
                "input": "hello",
                "tools": [{"type": "web_search"}],
            }
        )
    assert hosted.value.param == "tools[0].type"
    assert hosted.value.code is ErrorCode.UNSUPPORTED_OP

    with pytest.raises(OpenAIProtocolError) as undeclared:
        normalize_chat_completions_request(
            {
                "model": "ams-glm-4.7-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "bash"},
                },
            }
        )
    assert undeclared.value.param == "tool_choice"
