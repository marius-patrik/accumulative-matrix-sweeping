"""Fail-closed OpenAI request normalization for the local AMS boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class OpenAIEndpoint(StrEnum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class ContentKind(StrEnum):
    TEXT = "text"
    IMAGE_URL = "image_url"


@dataclass(frozen=True, slots=True)
class OpenAIRequestLimits:
    max_body_bytes: int = 16 * 1024 * 1024
    max_input_items: int = 16_384
    max_content_parts: int = 65_536
    max_tools: int = 256
    max_string_bytes: int = 8 * 1024 * 1024
    max_schema_bytes: int = 1024 * 1024
    max_output_tokens: int = 131_072

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be a positive integer")


class OpenAIProtocolError(AmsError):
    """An AMS error with a stable OpenAI-compatible HTTP projection."""

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        api_code: str = "invalid_value",
        error_type: str = "invalid_request_error",
        http_status: int = 400,
        retriable: bool = False,
        ams_code: ErrorCode = ErrorCode.PLAN_INVALID,
    ) -> None:
        super().__init__(
            ams_code,
            message,
            retriable=retriable,
            phase="preflight",
            subsystem="openai-api",
            evidence={"param": param} if param is not None else None,
        )
        self.param = param
        self.api_code = api_code
        self.error_type = error_type
        self.http_status = http_status


@dataclass(frozen=True, slots=True)
class ContentPart:
    kind: ContentKind
    text: str | None = None
    image_url: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MessageItem:
    role: str
    content: tuple[ContentPart, ...]


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    item_id: str | None
    summary: tuple[str, ...]
    content: tuple[str, ...]
    encrypted_content: str | None


@dataclass(frozen=True, slots=True)
class FunctionCallItem:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class FunctionOutputItem:
    call_id: str
    output: tuple[ContentPart, ...]


type InputItem = MessageItem | ReasoningItem | FunctionCallItem | FunctionOutputItem


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str | None
    parameters_json: bytes
    strict: bool | None


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: str
    function_name: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    kind: str
    name: str | None
    schema_json: bytes | None
    strict: bool | None


@dataclass(frozen=True, slots=True)
class NormalizedOpenAIRequest:
    endpoint: OpenAIEndpoint
    model: str
    stream: bool
    input_items: tuple[InputItem, ...]
    tools: tuple[FunctionTool, ...]
    tool_choice: ToolChoice | None
    structured_output: StructuredOutput | None
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    reasoning_effort: str | None
    reasoning_summary: str | None
    prompt_cache_key: str | None


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def parse_openai_json(
    payload: bytes,
    limits: OpenAIRequestLimits | None = None,
) -> dict[str, Any]:
    """Decode one bounded, duplicate-key-free JSON request object."""
    limits = limits or OpenAIRequestLimits()
    if len(payload) > limits.max_body_bytes:
        raise OpenAIProtocolError(
            "request body exceeds the configured byte limit",
            api_code="request_too_large",
            error_type="request_too_large",
            http_status=413,
        )
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        ValueError,
        RecursionError,
    ) as exc:
        raise OpenAIProtocolError(
            "request body is not strict UTF-8 JSON",
            api_code="invalid_json",
        ) from exc
    if not isinstance(value, dict):
        raise OpenAIProtocolError("request body must be a JSON object", api_code="invalid_json")
    return value


def _object(value: Any, param: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpenAIProtocolError(f"{param} must be an object", param=param)
    return value


def _array(value: Any, param: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise OpenAIProtocolError(f"{param} must be an array", param=param)
    if len(value) > maximum:
        raise OpenAIProtocolError(f"{param} exceeds the configured item limit", param=param)
    return value


def _string(
    value: Any,
    param: str,
    limits: OpenAIRequestLimits,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise OpenAIProtocolError(f"{param} must be a string", param=param)
    if len(value.encode("utf-8")) > limits.max_string_bytes:
        raise OpenAIProtocolError(f"{param} exceeds the configured byte limit", param=param)
    return value


def _optional_string(value: Any, param: str, limits: OpenAIRequestLimits) -> str | None:
    if value is None:
        return None
    return _string(value, param, limits, allow_empty=True)


def _exact_fields(value: dict[str, Any], allowed: set[str], param: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        field = f"{param}.{unknown[0]}" if param else unknown[0]
        raise OpenAIProtocolError(
            f"unsupported request field: {field}",
            param=field,
            api_code="unsupported_parameter",
            ams_code=ErrorCode.UNSUPPORTED_OP,
        )


def _optional_bool(value: Any, param: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise OpenAIProtocolError(f"{param} must be a boolean", param=param)
    return value


def _optional_number(value: Any, param: str, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OpenAIProtocolError(f"{param} must be a number", param=param)
    result = float(value)
    if not math.isfinite(result) or result < lower or result > upper:
        raise OpenAIProtocolError(f"{param} is outside the supported range", param=param)
    return result


def _max_tokens(value: Any, param: str, limits: OpenAIRequestLimits) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenAIProtocolError(f"{param} must be a positive integer", param=param)
    if value > limits.max_output_tokens:
        raise OpenAIProtocolError(f"{param} exceeds the configured token limit", param=param)
    return value


def _canonical_object(value: Any, param: str, limits: OpenAIRequestLimits) -> bytes:
    obj = _object(value, param)
    try:
        encoded = canonical_json_bytes(obj)
    except (AmsError, RecursionError) as exc:
        raise OpenAIProtocolError(f"{param} is not canonical JSON data", param=param) from exc
    if len(encoded) > limits.max_schema_bytes:
        raise OpenAIProtocolError(f"{param} exceeds the configured schema limit", param=param)
    return encoded


def _tool_name(value: Any, param: str, limits: OpenAIRequestLimits) -> str:
    name = _string(value, param, limits)
    if len(name) > 64 or _TOOL_NAME.fullmatch(name) is None:
        raise OpenAIProtocolError(f"{param} is not a valid function name", param=param)
    return name


def _text_part(text: Any, param: str, limits: OpenAIRequestLimits) -> ContentPart:
    return ContentPart(ContentKind.TEXT, text=_string(text, param, limits, allow_empty=True))


def _responses_content(
    value: Any,
    param: str,
    limits: OpenAIRequestLimits,
) -> tuple[ContentPart, ...]:
    if isinstance(value, str):
        return (_text_part(value, param, limits),)
    parts = _array(value, param, limits.max_content_parts)
    result: list[ContentPart] = []
    for index, raw in enumerate(parts):
        part_param = f"{param}[{index}]"
        part = _object(raw, part_param)
        part_type = part.get("type")
        if part_type in {"input_text", "output_text"}:
            _exact_fields(part, {"type", "text"}, part_param)
            result.append(_text_part(part.get("text"), f"{part_param}.text", limits))
        elif part_type == "input_image":
            _exact_fields(part, {"type", "image_url", "file_id", "detail"}, part_param)
            if part.get("file_id") is not None:
                raise OpenAIProtocolError(
                    "file-backed image input is not supported",
                    param=f"{part_param}.file_id",
                    api_code="unsupported_parameter",
                    ams_code=ErrorCode.UNSUPPORTED_OP,
                )
            url = _string(part.get("image_url"), f"{part_param}.image_url", limits)
            detail = _optional_string(part.get("detail"), f"{part_param}.detail", limits)
            if detail not in {None, "auto", "low", "high", "original"}:
                raise OpenAIProtocolError(
                    f"{part_param}.detail is unsupported", param=f"{part_param}.detail"
                )
            result.append(ContentPart(ContentKind.IMAGE_URL, image_url=url, detail=detail))
        else:
            raise OpenAIProtocolError(
                f"{part_param}.type is unsupported",
                param=f"{part_param}.type",
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )
    return tuple(result)


def _chat_content(
    value: Any,
    param: str,
    limits: OpenAIRequestLimits,
) -> tuple[ContentPart, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_text_part(value, param, limits),)
    parts = _array(value, param, limits.max_content_parts)
    result: list[ContentPart] = []
    for index, raw in enumerate(parts):
        part_param = f"{param}[{index}]"
        part = _object(raw, part_param)
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            _exact_fields(part, {"type", "text"}, part_param)
            result.append(_text_part(part.get("text"), f"{part_param}.text", limits))
        elif part_type in {"image_url", "input_image"}:
            _exact_fields(part, {"type", "image_url"}, part_param)
            image = part.get("image_url")
            if isinstance(image, str):
                url = _string(image, f"{part_param}.image_url", limits)
                detail = None
            else:
                image_obj = _object(image, f"{part_param}.image_url")
                _exact_fields(image_obj, {"url", "detail"}, f"{part_param}.image_url")
                url = _string(image_obj.get("url"), f"{part_param}.image_url.url", limits)
                detail = _optional_string(
                    image_obj.get("detail"), f"{part_param}.image_url.detail", limits
                )
            if detail not in {None, "auto", "low", "high", "original"}:
                raise OpenAIProtocolError(
                    f"{part_param}.image_url.detail is unsupported",
                    param=f"{part_param}.image_url.detail",
                )
            result.append(ContentPart(ContentKind.IMAGE_URL, image_url=url, detail=detail))
        else:
            raise OpenAIProtocolError(
                f"{part_param}.type is unsupported",
                param=f"{part_param}.type",
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )
    return tuple(result)


def _reasoning_texts(
    value: Any,
    param: str,
    expected_type: str,
    limits: OpenAIRequestLimits,
) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _array(value, param, limits.max_content_parts)
    result: list[str] = []
    for index, raw in enumerate(items):
        item_param = f"{param}[{index}]"
        item = _object(raw, item_param)
        _exact_fields(item, {"type", "text"}, item_param)
        if item.get("type") != expected_type:
            raise OpenAIProtocolError(
                f"{item_param}.type must be {expected_type}", param=f"{item_param}.type"
            )
        result.append(_string(item.get("text"), f"{item_param}.text", limits, allow_empty=True))
    return tuple(result)


def _responses_input(value: Any, limits: OpenAIRequestLimits) -> tuple[InputItem, ...]:
    if isinstance(value, str):
        return (MessageItem("user", (_text_part(value, "input", limits),)),)
    rows = _array(value, "input", limits.max_input_items)
    result: list[InputItem] = []
    for index, raw in enumerate(rows):
        param = f"input[{index}]"
        item = _object(raw, param)
        item_type = item.get("type", "message" if "role" in item else None)
        if item_type == "message":
            _exact_fields(item, {"type", "role", "content"}, param)
            role = _string(item.get("role"), f"{param}.role", limits)
            if role not in {"system", "developer", "user", "assistant"}:
                raise OpenAIProtocolError(f"{param}.role is unsupported", param=f"{param}.role")
            result.append(
                MessageItem(
                    role, _responses_content(item.get("content"), f"{param}.content", limits)
                )
            )
        elif item_type == "function_call":
            _exact_fields(item, {"type", "call_id", "name", "arguments", "id", "status"}, param)
            result.append(
                FunctionCallItem(
                    _string(item.get("call_id"), f"{param}.call_id", limits),
                    _tool_name(item.get("name"), f"{param}.name", limits),
                    _string(item.get("arguments"), f"{param}.arguments", limits, allow_empty=True),
                )
            )
        elif item_type == "function_call_output":
            _exact_fields(item, {"type", "call_id", "output", "id", "status"}, param)
            result.append(
                FunctionOutputItem(
                    _string(item.get("call_id"), f"{param}.call_id", limits),
                    _responses_content(item.get("output"), f"{param}.output", limits),
                )
            )
        elif item_type == "reasoning":
            _exact_fields(
                item,
                {"type", "id", "summary", "content", "encrypted_content", "status"},
                param,
            )
            result.append(
                ReasoningItem(
                    _optional_string(item.get("id"), f"{param}.id", limits),
                    _reasoning_texts(
                        item.get("summary"), f"{param}.summary", "summary_text", limits
                    ),
                    _reasoning_texts(
                        item.get("content"), f"{param}.content", "reasoning_text", limits
                    ),
                    _optional_string(
                        item.get("encrypted_content"), f"{param}.encrypted_content", limits
                    ),
                )
            )
        else:
            raise OpenAIProtocolError(
                f"{param}.type is unsupported",
                param=f"{param}.type",
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )
    if not result:
        raise OpenAIProtocolError("input must contain at least one item", param="input")
    return tuple(result)


def _chat_input(value: Any, limits: OpenAIRequestLimits) -> tuple[InputItem, ...]:
    rows = _array(value, "messages", limits.max_input_items)
    result: list[InputItem] = []
    for index, raw in enumerate(rows):
        param = f"messages[{index}]"
        message = _object(raw, param)
        role = _string(message.get("role"), f"{param}.role", limits)
        if role == "tool":
            _exact_fields(message, {"role", "content", "tool_call_id", "name"}, param)
            result.append(
                FunctionOutputItem(
                    _string(message.get("tool_call_id"), f"{param}.tool_call_id", limits),
                    _chat_content(message.get("content"), f"{param}.content", limits),
                )
            )
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise OpenAIProtocolError(f"{param}.role is unsupported", param=f"{param}.role")
        _exact_fields(
            message,
            {"role", "content", "name", "tool_calls", "reasoning_content", "refusal"},
            param,
        )
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            result.append(
                ReasoningItem(
                    None,
                    (),
                    (_string(reasoning, f"{param}.reasoning_content", limits, allow_empty=True),),
                    None,
                )
            )
        content = _chat_content(message.get("content"), f"{param}.content", limits)
        if content or role != "assistant" or not message.get("tool_calls"):
            result.append(MessageItem(role, content))
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            calls = _array(tool_calls, f"{param}.tool_calls", limits.max_tools)
            for call_index, raw_call in enumerate(calls):
                call_param = f"{param}.tool_calls[{call_index}]"
                call = _object(raw_call, call_param)
                _exact_fields(call, {"id", "type", "function"}, call_param)
                if call.get("type") != "function":
                    raise OpenAIProtocolError(
                        f"{call_param}.type is unsupported", param=f"{call_param}.type"
                    )
                function = _object(call.get("function"), f"{call_param}.function")
                _exact_fields(function, {"name", "arguments"}, f"{call_param}.function")
                result.append(
                    FunctionCallItem(
                        _string(call.get("id"), f"{call_param}.id", limits),
                        _tool_name(function.get("name"), f"{call_param}.function.name", limits),
                        _string(
                            function.get("arguments"),
                            f"{call_param}.function.arguments",
                            limits,
                            allow_empty=True,
                        ),
                    )
                )
    if not result:
        raise OpenAIProtocolError("messages must contain at least one item", param="messages")
    return tuple(result)


def _responses_tools(value: Any, limits: OpenAIRequestLimits) -> tuple[FunctionTool, ...]:
    if value is None:
        return ()
    rows = _array(value, "tools", limits.max_tools)
    result: list[FunctionTool] = []
    for index, raw in enumerate(rows):
        param = f"tools[{index}]"
        tool = _object(raw, param)
        if tool.get("type") != "function":
            raise OpenAIProtocolError(
                f"{param}.type is unsupported",
                param=f"{param}.type",
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )
        _exact_fields(tool, {"type", "name", "description", "parameters", "strict"}, param)
        result.append(
            FunctionTool(
                _tool_name(tool.get("name"), f"{param}.name", limits),
                _optional_string(tool.get("description"), f"{param}.description", limits),
                _canonical_object(tool.get("parameters", {}), f"{param}.parameters", limits),
                _optional_bool(tool.get("strict"), f"{param}.strict"),
            )
        )
    _validate_tool_names(result)
    return tuple(result)


def _chat_tools(value: Any, limits: OpenAIRequestLimits) -> tuple[FunctionTool, ...]:
    if value is None:
        return ()
    rows = _array(value, "tools", limits.max_tools)
    result: list[FunctionTool] = []
    for index, raw in enumerate(rows):
        param = f"tools[{index}]"
        tool = _object(raw, param)
        _exact_fields(tool, {"type", "function"}, param)
        if tool.get("type") != "function":
            raise OpenAIProtocolError(f"{param}.type is unsupported", param=f"{param}.type")
        function = _object(tool.get("function"), f"{param}.function")
        _exact_fields(
            function, {"name", "description", "parameters", "strict"}, f"{param}.function"
        )
        result.append(
            FunctionTool(
                _tool_name(function.get("name"), f"{param}.function.name", limits),
                _optional_string(
                    function.get("description"), f"{param}.function.description", limits
                ),
                _canonical_object(
                    function.get("parameters", {}), f"{param}.function.parameters", limits
                ),
                _optional_bool(function.get("strict"), f"{param}.function.strict"),
            )
        )
    _validate_tool_names(result)
    return tuple(result)


def _validate_tool_names(tools: list[FunctionTool]) -> None:
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise OpenAIProtocolError("tool names must be unique", param="tools")


def _tool_choice(
    value: Any, tools: tuple[FunctionTool, ...], limits: OpenAIRequestLimits
) -> ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise OpenAIProtocolError("tool_choice mode is unsupported", param="tool_choice")
        result = ToolChoice(value)
    else:
        choice = _object(value, "tool_choice")
        if choice.get("type") != "function":
            raise OpenAIProtocolError("tool_choice.type must be function", param="tool_choice.type")
        _exact_fields(choice, {"type", "name", "function"}, "tool_choice")
        if choice.get("function") is not None:
            function = _object(choice.get("function"), "tool_choice.function")
            _exact_fields(function, {"name"}, "tool_choice.function")
            raw_name = function.get("name")
            name_param = "tool_choice.function.name"
        else:
            raw_name = choice.get("name")
            name_param = "tool_choice.name"
        result = ToolChoice("function", _tool_name(raw_name, name_param, limits))
    if result.mode in {"required", "function"} and not tools:
        raise OpenAIProtocolError("tool_choice requires at least one tool", param="tool_choice")
    if result.function_name is not None and result.function_name not in {
        tool.name for tool in tools
    }:
        raise OpenAIProtocolError("tool_choice names an undeclared function", param="tool_choice")
    return result


def _responses_structured(value: Any, limits: OpenAIRequestLimits) -> StructuredOutput | None:
    if value is None:
        return None
    text = _object(value, "text")
    _exact_fields(text, {"format", "verbosity"}, "text")
    if text.get("verbosity") not in {None, "low", "medium", "high"}:
        raise OpenAIProtocolError("text.verbosity is unsupported", param="text.verbosity")
    raw_format = text.get("format")
    if raw_format is None:
        return None
    fmt = _object(raw_format, "text.format")
    kind = fmt.get("type")
    if kind == "text":
        _exact_fields(fmt, {"type"}, "text.format")
        return None
    if kind == "json_object":
        _exact_fields(fmt, {"type"}, "text.format")
        return StructuredOutput("json_object", None, None, None)
    if kind != "json_schema":
        raise OpenAIProtocolError("text.format.type is unsupported", param="text.format.type")
    _exact_fields(fmt, {"type", "name", "description", "schema", "strict"}, "text.format")
    return StructuredOutput(
        "json_schema",
        _string(fmt.get("name"), "text.format.name", limits),
        _canonical_object(fmt.get("schema"), "text.format.schema", limits),
        _optional_bool(fmt.get("strict"), "text.format.strict"),
    )


def _chat_structured(value: Any, limits: OpenAIRequestLimits) -> StructuredOutput | None:
    if value is None:
        return None
    fmt = _object(value, "response_format")
    kind = fmt.get("type")
    if kind == "text":
        _exact_fields(fmt, {"type"}, "response_format")
        return None
    if kind == "json_object":
        _exact_fields(fmt, {"type"}, "response_format")
        return StructuredOutput("json_object", None, None, None)
    if kind != "json_schema":
        raise OpenAIProtocolError(
            "response_format.type is unsupported", param="response_format.type"
        )
    _exact_fields(fmt, {"type", "json_schema"}, "response_format")
    schema = _object(fmt.get("json_schema"), "response_format.json_schema")
    _exact_fields(
        schema,
        {"name", "description", "schema", "strict"},
        "response_format.json_schema",
    )
    return StructuredOutput(
        "json_schema",
        _string(schema.get("name"), "response_format.json_schema.name", limits),
        _canonical_object(schema.get("schema"), "response_format.json_schema.schema", limits),
        _optional_bool(schema.get("strict"), "response_format.json_schema.strict"),
    )


def _reasoning(value: Any, limits: OpenAIRequestLimits) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    reasoning = _object(value, "reasoning")
    _exact_fields(reasoning, {"effort", "summary"}, "reasoning")
    effort = _optional_string(reasoning.get("effort"), "reasoning.effort", limits)
    if effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise OpenAIProtocolError("reasoning.effort is unsupported", param="reasoning.effort")
    summary = _optional_string(reasoning.get("summary"), "reasoning.summary", limits)
    if summary not in {None, "auto", "concise", "detailed"}:
        raise OpenAIProtocolError("reasoning.summary is unsupported", param="reasoning.summary")
    return effort, summary


def _reject_non_null(payload: dict[str, Any], fields: set[str]) -> None:
    for field in sorted(fields):
        if payload.get(field) is not None:
            raise OpenAIProtocolError(
                f"{field} is not supported by the local server",
                param=field,
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )


def normalize_responses_request(
    payload: dict[str, Any],
    limits: OpenAIRequestLimits | None = None,
) -> NormalizedOpenAIRequest:
    """Normalize a Responses request into the local model contract."""
    limits = limits or OpenAIRequestLimits()
    allowed = {
        "background",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "stream_tool_calls",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
    }
    _exact_fields(payload, allowed, "")
    _reject_non_null(
        payload,
        {
            "background",
            "conversation",
            "max_tool_calls",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_retention",
            "service_tier",
            "stream_options",
            "top_logprobs",
            "truncation",
        },
    )
    store = _optional_bool(payload.get("store"), "store")
    if store:
        raise OpenAIProtocolError(
            "store=true is unavailable on the stateless local server",
            param="store",
            api_code="unsupported_parameter",
            ams_code=ErrorCode.UNSUPPORTED_OP,
        )
    include = payload.get("include")
    if include is not None:
        includes = _array(include, "include", 32)
        for index, item in enumerate(includes):
            value = _string(item, f"include[{index}]", limits)
            if value != "reasoning.encrypted_content":
                raise OpenAIProtocolError(
                    f"include[{index}] is unsupported", param=f"include[{index}]"
                )
    stream = _optional_bool(payload.get("stream"), "stream") or False
    _optional_bool(payload.get("stream_tool_calls"), "stream_tool_calls")
    instructions = payload.get("instructions")
    input_items = list(_responses_input(payload.get("input"), limits))
    if instructions is not None:
        input_items.insert(
            0,
            MessageItem(
                "developer",
                (_text_part(instructions, "instructions", limits),),
            ),
        )
    tools = _responses_tools(payload.get("tools"), limits)
    effort, summary = _reasoning(payload.get("reasoning"), limits)
    return NormalizedOpenAIRequest(
        endpoint=OpenAIEndpoint.RESPONSES,
        model=_string(payload.get("model"), "model", limits),
        stream=stream,
        input_items=tuple(input_items),
        tools=tools,
        tool_choice=_tool_choice(payload.get("tool_choice"), tools, limits),
        structured_output=_responses_structured(payload.get("text"), limits),
        max_output_tokens=_max_tokens(
            payload.get("max_output_tokens"), "max_output_tokens", limits
        ),
        temperature=_optional_number(payload.get("temperature"), "temperature", 0.0, 2.0),
        top_p=_optional_number(payload.get("top_p"), "top_p", 0.0, 1.0),
        reasoning_effort=effort,
        reasoning_summary=summary,
        prompt_cache_key=_optional_string(
            payload.get("prompt_cache_key"), "prompt_cache_key", limits
        ),
    )


def normalize_chat_completions_request(
    payload: dict[str, Any],
    limits: OpenAIRequestLimits | None = None,
) -> NormalizedOpenAIRequest:
    """Normalize a Chat Completions request into the same local model contract."""
    limits = limits or OpenAIRequestLimits()
    allowed = {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "search_parameters",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
    }
    _exact_fields(payload, allowed, "")
    _reject_non_null(
        payload,
        {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "parallel_tool_calls",
            "presence_penalty",
            "seed",
            "search_parameters",
            "service_tier",
            "stop",
            "top_logprobs",
        },
    )
    if payload.get("n") not in {None, 1}:
        raise OpenAIProtocolError("only n=1 is supported", param="n")
    if _optional_bool(payload.get("store"), "store"):
        raise OpenAIProtocolError(
            "store=true is unavailable on the stateless local server",
            param="store",
            api_code="unsupported_parameter",
            ams_code=ErrorCode.UNSUPPORTED_OP,
        )
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        options = _object(stream_options, "stream_options")
        _exact_fields(options, {"include_usage", "include_obfuscation"}, "stream_options")
        _optional_bool(options.get("include_usage"), "stream_options.include_usage")
        _optional_bool(options.get("include_obfuscation"), "stream_options.include_obfuscation")
    max_tokens = payload.get("max_completion_tokens")
    max_param = "max_completion_tokens"
    if max_tokens is None:
        max_tokens = payload.get("max_tokens")
        max_param = "max_tokens"
    elif payload.get("max_tokens") is not None:
        raise OpenAIProtocolError(
            "max_tokens and max_completion_tokens are mutually exclusive",
            param="max_tokens",
        )
    tools = _chat_tools(payload.get("tools"), limits)
    effort = _optional_string(payload.get("reasoning_effort"), "reasoning_effort", limits)
    if effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise OpenAIProtocolError("reasoning_effort is unsupported", param="reasoning_effort")
    return NormalizedOpenAIRequest(
        endpoint=OpenAIEndpoint.CHAT_COMPLETIONS,
        model=_string(payload.get("model"), "model", limits),
        stream=_optional_bool(payload.get("stream"), "stream") or False,
        input_items=_chat_input(payload.get("messages"), limits),
        tools=tools,
        tool_choice=_tool_choice(payload.get("tool_choice"), tools, limits),
        structured_output=_chat_structured(payload.get("response_format"), limits),
        max_output_tokens=_max_tokens(max_tokens, max_param, limits),
        temperature=_optional_number(payload.get("temperature"), "temperature", 0.0, 2.0),
        top_p=_optional_number(payload.get("top_p"), "top_p", 0.0, 1.0),
        reasoning_effort=effort,
        reasoning_summary=None,
        prompt_cache_key=None,
    )
