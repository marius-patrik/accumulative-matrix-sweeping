"""Typed model-event contract and transactional generation accumulator."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Event
from typing import Protocol

from ams.api.contracts import NormalizedOpenAIRequest
from ams.errors import AmsError, ErrorCode


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    f"generation usage {name} must be a nonnegative integer",
                    subsystem="openai-api",
                )
        if self.cached_input_tokens > self.input_tokens:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "cached input tokens exceed input tokens",
                subsystem="openai-api",
            )
        if self.reasoning_tokens > self.output_tokens:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "reasoning tokens exceed output tokens",
                subsystem="openai-api",
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    index: int
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolArgumentsDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class GenerationCompleted:
    usage: GenerationUsage
    finish_reason: str = "stop"


type GenerationEvent = (
    TextDelta | ReasoningDelta | ToolCallStart | ToolArgumentsDelta | GenerationCompleted
)


class InferenceBackend(Protocol):
    """A local engine that yields bounded, ordered generation events."""

    def stream(
        self,
        request: NormalizedOpenAIRequest,
        cancellation: Event,
    ) -> Iterable[GenerationEvent]: ...


@dataclass(frozen=True, slots=True)
class GeneratedToolCall:
    index: int
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    text: str
    reasoning: str
    tool_calls: tuple[GeneratedToolCall, ...]
    output_order: tuple[tuple[str, int], ...]
    usage: GenerationUsage
    finish_reason: str


@dataclass(slots=True)
class _ToolState:
    call_id: str
    name: str
    arguments: list[str]


def _backend_invariant(message: str) -> AmsError:
    return AmsError(
        ErrorCode.INTERNAL_INVARIANT,
        message,
        phase="decode",
        subsystem="openai-api",
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _strict_json_value(value: str, description: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise _backend_invariant(f"{description} is not strict JSON") from exc


def _validate_arguments(arguments: str, index: int) -> None:
    value = _strict_json_value(arguments, f"tool call {index} arguments")
    if not isinstance(value, dict):
        raise _backend_invariant(f"tool call {index} arguments must encode an object")


class GenerationAccumulator:
    """Validate one backend stream and publish a final snapshot only on completion."""

    def __init__(self, request: NormalizedOpenAIRequest):
        self.request = request
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tools: dict[int, _ToolState] = {}
        self._output_order: list[tuple[str, int]] = []
        self._text_started = False
        self._reasoning_started = False
        self._snapshot: GenerationSnapshot | None = None

    @property
    def completed(self) -> bool:
        return self._snapshot is not None

    @property
    def text(self) -> str:
        return "".join(self._text)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning)

    @property
    def output_order(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._output_order)

    def tool_call(self, index: int) -> GeneratedToolCall:
        state = self._tools.get(index)
        if state is None:
            raise _backend_invariant(f"tool call {index} has not started")
        return GeneratedToolCall(index, state.call_id, state.name, "".join(state.arguments))

    def accept(self, event: GenerationEvent) -> None:
        if self.completed:
            raise _backend_invariant("backend emitted an event after completion")
        if isinstance(event, TextDelta):
            if not isinstance(event.text, str) or not event.text:
                raise _backend_invariant("backend emitted an empty text delta")
            if not self._text_started:
                self._text_started = True
                self._output_order.append(("message", 0))
            self._text.append(event.text)
            return
        if isinstance(event, ReasoningDelta):
            if not isinstance(event.text, str) or not event.text:
                raise _backend_invariant("backend emitted an empty reasoning delta")
            if not self._reasoning_started:
                self._reasoning_started = True
                self._output_order.append(("reasoning", 0))
            self._reasoning.append(event.text)
            return
        if isinstance(event, ToolCallStart):
            if (
                isinstance(event.index, bool)
                or not isinstance(event.index, int)
                or event.index != len(self._tools)
            ):
                raise _backend_invariant("tool call indices must be contiguous and source ordered")
            if (
                not isinstance(event.call_id, str)
                or not event.call_id
                or not isinstance(event.name, str)
                or not event.name
            ):
                raise _backend_invariant("tool call identity is empty")
            if event.call_id in {state.call_id for state in self._tools.values()}:
                raise _backend_invariant("tool call IDs must be unique")
            if event.name not in {tool.name for tool in self.request.tools}:
                raise _backend_invariant("backend selected an undeclared tool")
            self._tools[event.index] = _ToolState(event.call_id, event.name, [])
            self._output_order.append(("tool", event.index))
            return
        if isinstance(event, ToolArgumentsDelta):
            if not isinstance(event.delta, str) or not event.delta:
                raise _backend_invariant("backend emitted an empty tool argument delta")
            state = self._tools.get(event.index)
            if state is None:
                raise _backend_invariant("tool arguments arrived before the tool call started")
            state.arguments.append(event.delta)
            return
        if not isinstance(event, GenerationCompleted):
            raise _backend_invariant("backend emitted an unknown generation event")
        allowed_finish_reasons = {"stop", "length", "content_filter", "tool_calls"}
        if event.finish_reason not in allowed_finish_reasons:
            raise _backend_invariant("backend emitted an unsupported finish reason")
        if self._tools and event.finish_reason != "tool_calls":
            raise _backend_invariant("a tool-bearing generation must finish with tool_calls")
        if not self._tools and event.finish_reason == "tool_calls":
            raise _backend_invariant("tool_calls finish reason has no tool call")
        if (
            self.request.max_output_tokens is not None
            and event.usage.output_tokens > self.request.max_output_tokens
        ):
            raise _backend_invariant("backend usage exceeds the requested output-token limit")
        tools = tuple(self.tool_call(index) for index in range(len(self._tools)))
        for tool in tools:
            _validate_arguments(tool.arguments, tool.index)
        text = self.text
        if self.request.structured_output is not None and event.finish_reason != "tool_calls":
            structured = _strict_json_value(text, "structured output")
            if self.request.structured_output.kind == "json_object" and not isinstance(
                structured, dict
            ):
                raise _backend_invariant("json_object output is not an object")
        if not self._output_order:
            self._text_started = True
            self._output_order.append(("message", 0))
        self._snapshot = GenerationSnapshot(
            text=text,
            reasoning=self.reasoning,
            tool_calls=tools,
            output_order=tuple(self._output_order),
            usage=event.usage,
            finish_reason=event.finish_reason,
        )

    def snapshot(self) -> GenerationSnapshot:
        if self._snapshot is None:
            raise _backend_invariant("backend stream ended without completion")
        return self._snapshot
