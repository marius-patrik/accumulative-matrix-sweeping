"""OpenAI Responses and Chat Completions encoders over one model-event stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ams.api.contracts import NormalizedOpenAIRequest, OpenAIEndpoint
from ams.api.generation import (
    GenerationAccumulator,
    GenerationCompleted,
    GenerationEvent,
    GenerationSnapshot,
    ReasoningDelta,
    TextDelta,
    ToolArgumentsDelta,
    ToolCallStart,
)
from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode


@dataclass(frozen=True, slots=True)
class ResponseIdentity:
    response_id: str
    message_id: str
    reasoning_id: str
    created_at: int

    def __post_init__(self) -> None:
        for name in ("response_id", "message_id", "reasoning_id"):
            if not getattr(self, name):
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    f"{name} must be nonempty",
                    subsystem="openai-api",
                )
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, int):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "created_at must be an integer",
                subsystem="openai-api",
            )


def sse_data(payload: dict[str, Any] | str) -> bytes:
    data = payload.encode("utf-8") if isinstance(payload, str) else canonical_json_bytes(payload)
    return b"data: " + data + b"\n\n"


class OpenAIStreamSession:
    """Encode a validated backend stream while retaining one transactional final response."""

    def __init__(self, request: NormalizedOpenAIRequest, identity: ResponseIdentity):
        self.request = request
        self.identity = identity
        self.accumulator = GenerationAccumulator(request)
        self._sequence = 0
        self._message_open = False
        self._reasoning_open = False
        self._chat_role_sent = False

    def _responses_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        payload = {"type": event_type, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        return payload

    def start(self) -> tuple[dict[str, Any], ...]:
        if self.request.endpoint is OpenAIEndpoint.CHAT_COMPLETIONS:
            return ()
        return (
            self._responses_event(
                "response.created",
                response=self._responses_object(status="in_progress", output=[], usage=None),
            ),
        )

    def push(self, event: GenerationEvent) -> tuple[dict[str, Any], ...]:
        if self.request.endpoint is OpenAIEndpoint.RESPONSES:
            return self._push_responses(event)
        return self._push_chat(event)

    def error_event(self, error: AmsError) -> dict[str, Any]:
        """Encode a terminal error after streaming headers have already been sent."""
        code = error.code.value.lower()
        if self.request.endpoint is OpenAIEndpoint.RESPONSES:
            return self._responses_event(
                "error",
                code=code,
                message=error.message,
                param=getattr(error, "param", None),
            )
        return {
            "error": {
                "message": error.message,
                "type": getattr(error, "error_type", "server_error"),
                "param": getattr(error, "param", None),
                "code": getattr(error, "api_code", code),
            }
        }

    def _open_message(self) -> list[dict[str, Any]]:
        if self._message_open:
            return []
        self._message_open = True
        output_index = self.accumulator.output_order.index(("message", 0))
        item = self._message_item(self.accumulator.text, status="in_progress")
        item["content"] = []
        return [
            self._responses_event(
                "response.output_item.added", output_index=output_index, item=item
            ),
            self._responses_event(
                "response.content_part.added",
                item_id=self.identity.message_id,
                output_index=output_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            ),
        ]

    def _open_reasoning(self) -> list[dict[str, Any]]:
        if self._reasoning_open:
            return []
        self._reasoning_open = True
        output_index = self.accumulator.output_order.index(("reasoning", 0))
        return [
            self._responses_event(
                "response.output_item.added",
                output_index=output_index,
                item={
                    "type": "reasoning",
                    "id": self.identity.reasoning_id,
                    "summary": [],
                    "status": "in_progress",
                },
            ),
            self._responses_event(
                "response.reasoning_summary_part.added",
                item_id=self.identity.reasoning_id,
                output_index=output_index,
                summary_index=0,
                part={"type": "summary_text", "text": ""},
            ),
        ]

    def _tool_item_id(self, index: int) -> str:
        suffix = self.identity.response_id.removeprefix("resp_")
        return f"fc_{suffix}_{index}"

    def _push_responses(self, event: GenerationEvent) -> tuple[dict[str, Any], ...]:
        self.accumulator.accept(event)
        emitted: list[dict[str, Any]] = []
        if isinstance(event, TextDelta):
            emitted.extend(self._open_message())
            output_index = self.accumulator.output_order.index(("message", 0))
            emitted.append(
                self._responses_event(
                    "response.output_text.delta",
                    item_id=self.identity.message_id,
                    output_index=output_index,
                    content_index=0,
                    delta=event.text,
                )
            )
        elif isinstance(event, ReasoningDelta):
            emitted.extend(self._open_reasoning())
            output_index = self.accumulator.output_order.index(("reasoning", 0))
            emitted.append(
                self._responses_event(
                    "response.reasoning_summary_text.delta",
                    item_id=self.identity.reasoning_id,
                    output_index=output_index,
                    summary_index=0,
                    delta=event.text,
                )
            )
        elif isinstance(event, ToolCallStart):
            output_index = self.accumulator.output_order.index(("tool", event.index))
            emitted.append(
                self._responses_event(
                    "response.output_item.added",
                    output_index=output_index,
                    item={
                        "type": "function_call",
                        "id": self._tool_item_id(event.index),
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                )
            )
        elif isinstance(event, ToolArgumentsDelta):
            output_index = self.accumulator.output_order.index(("tool", event.index))
            emitted.append(
                self._responses_event(
                    "response.function_call_arguments.delta",
                    item_id=self._tool_item_id(event.index),
                    output_index=output_index,
                    delta=event.delta,
                )
            )
        elif isinstance(event, GenerationCompleted):
            emitted.extend(self._responses_completion_events())
        return tuple(emitted)

    def _responses_completion_events(self) -> list[dict[str, Any]]:
        snapshot = self.accumulator.snapshot()
        emitted: list[dict[str, Any]] = []
        if not self._message_open and ("message", 0) in snapshot.output_order:
            emitted.extend(self._open_message())
        for output_index, (kind, index) in enumerate(snapshot.output_order):
            if kind == "message":
                part = {
                    "type": "output_text",
                    "text": snapshot.text,
                    "annotations": [],
                }
                emitted.extend(
                    [
                        self._responses_event(
                            "response.output_text.done",
                            item_id=self.identity.message_id,
                            output_index=output_index,
                            content_index=0,
                            text=snapshot.text,
                        ),
                        self._responses_event(
                            "response.content_part.done",
                            item_id=self.identity.message_id,
                            output_index=output_index,
                            content_index=0,
                            part=part,
                        ),
                        self._responses_event(
                            "response.output_item.done",
                            output_index=output_index,
                            item=self._message_item(snapshot.text, status="completed"),
                        ),
                    ]
                )
            elif kind == "reasoning":
                summary_part = {"type": "summary_text", "text": snapshot.reasoning}
                emitted.extend(
                    [
                        self._responses_event(
                            "response.reasoning_summary_text.done",
                            item_id=self.identity.reasoning_id,
                            output_index=output_index,
                            summary_index=0,
                            text=snapshot.reasoning,
                        ),
                        self._responses_event(
                            "response.reasoning_summary_part.done",
                            item_id=self.identity.reasoning_id,
                            output_index=output_index,
                            summary_index=0,
                            part=summary_part,
                        ),
                        self._responses_event(
                            "response.output_item.done",
                            output_index=output_index,
                            item={
                                "type": "reasoning",
                                "id": self.identity.reasoning_id,
                                "summary": [summary_part],
                                "status": "completed",
                            },
                        ),
                    ]
                )
            else:
                tool = self.accumulator.tool_call(index)
                item = self._function_item(tool, status="completed")
                emitted.extend(
                    [
                        self._responses_event(
                            "response.function_call_arguments.done",
                            item_id=self._tool_item_id(index),
                            output_index=output_index,
                            name=tool.name,
                            arguments=tool.arguments,
                        ),
                        self._responses_event(
                            "response.output_item.done", output_index=output_index, item=item
                        ),
                    ]
                )
        emitted.append(
            self._responses_event(
                "response.completed",
                response=self._responses_object(
                    status="completed",
                    output=self._responses_output(snapshot),
                    usage=self._responses_usage(snapshot),
                ),
            )
        )
        return emitted

    def _chat_delta(
        self, delta: dict[str, Any], finish_reason: str | None, usage: Any = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.identity.response_id,
            "object": "chat.completion.chunk",
            "created": self.identity.created_at,
            "model": self.request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _with_chat_role(self, delta: dict[str, Any]) -> dict[str, Any]:
        if not self._chat_role_sent:
            self._chat_role_sent = True
            return {"role": "assistant", **delta}
        return delta

    def _push_chat(self, event: GenerationEvent) -> tuple[dict[str, Any], ...]:
        self.accumulator.accept(event)
        if isinstance(event, TextDelta):
            return (self._chat_delta(self._with_chat_role({"content": event.text}), None),)
        if isinstance(event, ReasoningDelta):
            return (
                self._chat_delta(self._with_chat_role({"reasoning_content": event.text}), None),
            )
        if isinstance(event, ToolCallStart):
            return (
                self._chat_delta(
                    self._with_chat_role(
                        {
                            "tool_calls": [
                                {
                                    "index": event.index,
                                    "id": event.call_id,
                                    "type": "function",
                                    "function": {"name": event.name, "arguments": ""},
                                }
                            ]
                        }
                    ),
                    None,
                ),
            )
        if isinstance(event, ToolArgumentsDelta):
            return (
                self._chat_delta(
                    {
                        "tool_calls": [
                            {
                                "index": event.index,
                                "function": {"arguments": event.delta},
                            }
                        ]
                    },
                    None,
                ),
            )
        snapshot = self.accumulator.snapshot()
        return (
            self._chat_delta(
                self._with_chat_role({}),
                snapshot.finish_reason,
                self._chat_usage(snapshot),
            ),
        )

    def response_body(self) -> dict[str, Any]:
        snapshot = self.accumulator.snapshot()
        if self.request.endpoint is OpenAIEndpoint.RESPONSES:
            return self._responses_object(
                status="completed",
                output=self._responses_output(snapshot),
                usage=self._responses_usage(snapshot),
            )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": snapshot.text or None,
        }
        if snapshot.reasoning:
            message["reasoning_content"] = snapshot.reasoning
        if snapshot.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool.call_id,
                    "type": "function",
                    "function": {"name": tool.name, "arguments": tool.arguments},
                }
                for tool in snapshot.tool_calls
            ]
        return {
            "id": self.identity.response_id,
            "object": "chat.completion",
            "created": self.identity.created_at,
            "model": self.request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": snapshot.finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": self._chat_usage(snapshot),
        }

    def _message_item(self, text: str, *, status: str) -> dict[str, Any]:
        return {
            "type": "message",
            "id": self.identity.message_id,
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def _function_item(self, tool: Any, *, status: str) -> dict[str, Any]:
        return {
            "type": "function_call",
            "id": self._tool_item_id(tool.index),
            "call_id": tool.call_id,
            "name": tool.name,
            "arguments": tool.arguments,
            "status": status,
        }

    def _responses_output(self, snapshot: GenerationSnapshot) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for kind, index in snapshot.output_order:
            if kind == "message":
                output.append(self._message_item(snapshot.text, status="completed"))
            elif kind == "reasoning":
                output.append(
                    {
                        "type": "reasoning",
                        "id": self.identity.reasoning_id,
                        "summary": [{"type": "summary_text", "text": snapshot.reasoning}],
                        "status": "completed",
                    }
                )
            else:
                output.append(
                    self._function_item(self.accumulator.tool_call(index), status="completed")
                )
        return output

    def _responses_object(
        self,
        *,
        status: str,
        output: list[dict[str, Any]],
        usage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.identity.response_id,
            "object": "response",
            "created_at": self.identity.created_at,
            "model": self.request.model,
            "status": status,
            "output": output,
        }
        if usage is not None:
            result["usage"] = usage
        return result

    @staticmethod
    def _responses_usage(snapshot: GenerationSnapshot) -> dict[str, Any]:
        usage = snapshot.usage
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": {"cached_tokens": usage.cached_input_tokens},
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        }

    @staticmethod
    def _chat_usage(snapshot: GenerationSnapshot) -> dict[str, Any]:
        usage = snapshot.usage
        return {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
            "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        }
