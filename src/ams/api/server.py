"""Bounded stdlib HTTP adapter for the local OpenAI-compatible AMS boundary."""

from __future__ import annotations

import math
import secrets
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore, Event
from typing import Any
from urllib.parse import urlsplit

from ams.api.contracts import (
    ContentKind,
    FunctionOutputItem,
    MessageItem,
    NormalizedOpenAIRequest,
    OpenAIEndpoint,
    OpenAIProtocolError,
    OpenAIRequestLimits,
    normalize_chat_completions_request,
    normalize_responses_request,
    parse_openai_json,
)
from ams.api.generation import GenerationCompleted, GenerationEvent, InferenceBackend
from ams.api.openai_stream import OpenAIStreamSession, ResponseIdentity, sse_data
from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode


@dataclass(frozen=True, slots=True)
class OpenAIServerConfig:
    models: tuple[str, ...]
    max_concurrent_requests: int = 1
    socket_timeout_seconds: float = 30.0
    supports_image_input: bool = False
    supports_json_schema: bool = False
    api_key: str | None = None
    request_limits: OpenAIRequestLimits = field(default_factory=OpenAIRequestLimits)

    def __post_init__(self) -> None:
        if not self.models or any(not isinstance(model, str) or not model for model in self.models):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "at least one nonempty model name is required",
                subsystem="openai-api",
            )
        if len(self.models) != len(set(self.models)):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "model names must be unique",
                subsystem="openai-api",
            )
        if (
            isinstance(self.max_concurrent_requests, bool)
            or not isinstance(self.max_concurrent_requests, int)
            or self.max_concurrent_requests <= 0
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "max_concurrent_requests must be a positive integer",
                subsystem="openai-api",
            )
        if (
            isinstance(self.socket_timeout_seconds, bool)
            or not isinstance(self.socket_timeout_seconds, int | float)
            or not math.isfinite(self.socket_timeout_seconds)
            or self.socket_timeout_seconds <= 0
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "socket_timeout_seconds must be positive",
                subsystem="openai-api",
            )
        if self.api_key is not None and not self.api_key:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "api_key cannot be empty",
                subsystem="openai-api",
            )
        if not isinstance(self.supports_image_input, bool) or not isinstance(
            self.supports_json_schema, bool
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "server capability flags must be booleans",
                subsystem="openai-api",
            )


@dataclass(frozen=True, slots=True)
class ApplicationResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes | ManagedStream


class ManagedStream(Iterator[bytes]):
    """Own one admission slot and make cancellation/release exactly-once."""

    def __init__(
        self,
        chunks: Iterator[bytes],
        cancellation: Event,
        release: Callable[[], None],
    ) -> None:
        self._chunks = chunks
        self.cancellation = cancellation
        self._release = release
        self._closed = False

    def __iter__(self) -> ManagedStream:
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        try:
            return next(self._chunks)
        except StopIteration:
            self.close()
            raise
        except Exception:
            self.close()
            raise

    def close(self, *, cancelled: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if cancelled:
            self.cancellation.set()
        close = getattr(self._chunks, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._release()


def _api_error(
    error: AmsError,
) -> tuple[int, str, str, str | None, str, bool]:
    if isinstance(error, OpenAIProtocolError):
        return (
            error.http_status,
            error.error_type,
            error.api_code,
            error.param,
            error.message,
            error.retriable,
        )
    if error.code in {ErrorCode.UNSUPPORTED_OP, ErrorCode.CAPABILITY_MISMATCH}:
        return 400, "invalid_request_error", "unsupported_parameter", None, error.message, False
    if error.code in {
        ErrorCode.PREFLIGHT_NO_WORKING_SET,
        ErrorCode.RESERVATION_LOST,
        ErrorCode.BROKER_VIOLATION,
    }:
        return 503, "server_error", "resource_unavailable", None, error.message, True
    if error.code is ErrorCode.CANCELLED:
        return 499, "server_error", "cancelled", None, error.message, False
    return (
        500,
        "server_error",
        error.code.value.lower(),
        None,
        error.message,
        error.retriable,
    )


class OpenAIApplication:
    """Normalize requests, admit one bounded engine run, and encode its result."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: OpenAIServerConfig,
        *,
        clock: Callable[[], int] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._admission = BoundedSemaphore(config.max_concurrent_requests)
        self._clock = clock or (lambda: int(time.time()))
        self._token_factory = token_factory or (lambda: secrets.token_hex(12))

    def dispatch(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        body: bytes = b"",
    ) -> ApplicationResponse:
        request_id = self._request_id()
        try:
            normalized_headers = {key.lower(): value for key, value in headers.items()}
            self._authenticate(normalized_headers)
            parsed = urlsplit(target)
            if parsed.query or parsed.fragment:
                raise OpenAIProtocolError(
                    "query strings and fragments are not supported",
                    api_code="unsupported_parameter",
                    ams_code=ErrorCode.UNSUPPORTED_OP,
                )
            if method == "GET" and parsed.path == "/healthz":
                return self._json_response(
                    200,
                    {"status": "ok", "models": list(self.config.models)},
                    request_id,
                )
            if method == "GET" and parsed.path == "/v1/models":
                return self._json_response(200, self._models_body(), request_id)
            if method != "POST":
                raise OpenAIProtocolError(
                    "route not found",
                    api_code="not_found",
                    error_type="not_found_error",
                    http_status=404,
                )
            content_type = normalized_headers.get("content-type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise OpenAIProtocolError(
                    "Content-Type must be application/json",
                    api_code="unsupported_media_type",
                    http_status=415,
                )
            payload = parse_openai_json(body, self.config.request_limits)
            if parsed.path == "/v1/responses":
                request = normalize_responses_request(payload, self.config.request_limits)
            elif parsed.path == "/v1/chat/completions":
                request = normalize_chat_completions_request(payload, self.config.request_limits)
            else:
                raise OpenAIProtocolError(
                    "route not found",
                    api_code="not_found",
                    error_type="not_found_error",
                    http_status=404,
                )
            if request.model not in self.config.models:
                raise OpenAIProtocolError(
                    f"model is not available: {request.model}",
                    param="model",
                    api_code="model_not_found",
                    http_status=404,
                )
            self._validate_capabilities(request)
            if not self._admission.acquire(blocking=False):
                raise OpenAIProtocolError(
                    "the configured local inference slot is busy",
                    api_code="rate_limit_exceeded",
                    error_type="rate_limit_error",
                    http_status=429,
                    retriable=True,
                    ams_code=ErrorCode.PREFLIGHT_NO_WORKING_SET,
                )
            return self._run_admitted(request, request_id)
        except AmsError as error:
            return self.error_response(error, request_id)
        except Exception as exc:
            error = AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "unexpected OpenAI boundary failure",
                subsystem="openai-api",
            )
            error.__cause__ = exc
            return self.error_response(error, request_id)

    def _run_admitted(
        self,
        request: NormalizedOpenAIRequest,
        request_id: str,
    ) -> ApplicationResponse:
        try:
            identity = self._identity(request.endpoint)
            session = OpenAIStreamSession(request, identity)
            cancellation = Event()
        except Exception:
            self._admission.release()
            raise
        if request.stream:
            chunks = self._stream_chunks(request, session, cancellation)
            managed = ManagedStream(chunks, cancellation, self._admission.release)
            return ApplicationResponse(
                200,
                (
                    ("Content-Type", "text/event-stream; charset=utf-8"),
                    ("Cache-Control", "no-cache"),
                    ("X-Accel-Buffering", "no"),
                    ("X-Request-Id", request_id),
                ),
                managed,
            )
        try:
            for event in self._validated_backend_events(request, cancellation):
                session.push(event)
            response = session.response_body()
            return self._json_response(200, response, request_id)
        except AmsError as error:
            return self.error_response(error, request_id)
        except Exception as exc:
            error = AmsError(
                ErrorCode.BACKEND_FAILURE,
                "local inference backend failed",
                retriable=True,
                phase="decode",
                subsystem="openai-api",
            )
            error.__cause__ = exc
            return self.error_response(error, request_id)
        finally:
            self._admission.release()

    def _validated_backend_events(
        self,
        request: NormalizedOpenAIRequest,
        cancellation: Event,
    ) -> Iterator[GenerationEvent]:
        iterator = iter(self.backend.stream(request, cancellation))
        try:
            pending = next(iterator)
        except StopIteration:
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "local inference backend returned an empty stream",
                retriable=True,
                phase="decode",
                subsystem="openai-api",
            ) from None
        while True:
            if cancellation.is_set():
                raise AmsError(
                    ErrorCode.CANCELLED,
                    "request was cancelled",
                    phase="decode",
                    subsystem="openai-api",
                )
            if isinstance(pending, GenerationCompleted):
                try:
                    next(iterator)
                except StopIteration:
                    yield pending
                    break
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    "backend emitted an event after completion",
                    phase="decode",
                    subsystem="openai-api",
                )
            yield pending
            try:
                pending = next(iterator)
            except StopIteration:
                break

    def _stream_chunks(
        self,
        request: NormalizedOpenAIRequest,
        session: OpenAIStreamSession,
        cancellation: Event,
    ) -> Iterator[bytes]:
        try:
            for payload in session.start():
                yield sse_data(payload)
            for event in self._validated_backend_events(request, cancellation):
                for payload in session.push(event):
                    yield sse_data(payload)
            session.response_body()
            yield sse_data("[DONE]")
        except AmsError as error:
            if not cancellation.is_set():
                yield sse_data(session.error_event(error))
        except Exception as exc:
            if not cancellation.is_set():
                error = AmsError(
                    ErrorCode.BACKEND_FAILURE,
                    "local inference backend failed",
                    retriable=True,
                    phase="decode",
                    subsystem="openai-api",
                )
                error.__cause__ = exc
                yield sse_data(session.error_event(error))

    def _authenticate(self, headers: Mapping[str, str]) -> None:
        expected = self.config.api_key
        if expected is None:
            return
        authorization = headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(token, expected)
        ):
            raise OpenAIProtocolError(
                "invalid bearer token",
                api_code="invalid_api_key",
                error_type="authentication_error",
                http_status=401,
            )

    def _validate_capabilities(self, request: NormalizedOpenAIRequest) -> None:
        if (
            request.structured_output is not None
            and request.structured_output.kind == "json_schema"
            and not self.config.supports_json_schema
        ):
            raise OpenAIProtocolError(
                "JSON Schema enforcement is not qualified by this backend",
                param=(
                    "text.format"
                    if request.endpoint is OpenAIEndpoint.RESPONSES
                    else "response_format"
                ),
                api_code="unsupported_parameter",
                ams_code=ErrorCode.UNSUPPORTED_OP,
            )
        if self.config.supports_image_input:
            return
        for index, item in enumerate(request.input_items):
            if not isinstance(item, MessageItem | FunctionOutputItem):
                continue
            if any(part.kind is ContentKind.IMAGE_URL for part in item.content):
                source = "input" if request.endpoint is OpenAIEndpoint.RESPONSES else "messages"
                raise OpenAIProtocolError(
                    "image input is not qualified by this backend",
                    param=f"{source}[{index}]",
                    api_code="unsupported_parameter",
                    ams_code=ErrorCode.UNSUPPORTED_OP,
                )

    def _identity(self, endpoint: OpenAIEndpoint) -> ResponseIdentity:
        token = self._token_factory()
        prefix = "resp" if endpoint is OpenAIEndpoint.RESPONSES else "chatcmpl"
        return ResponseIdentity(
            f"{prefix}_{token}",
            f"msg_{token}",
            f"rs_{token}",
            self._clock(),
        )

    def _request_id(self) -> str:
        return f"req_{self._token_factory()}"

    def _models_body(self) -> dict[str, Any]:
        created = self._clock()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": created,
                    "owned_by": "ams-local",
                }
                for model in self.config.models
            ],
        }

    def _json_response(
        self,
        status: int,
        payload: dict[str, Any],
        request_id: str,
        *,
        retriable: bool = False,
    ) -> ApplicationResponse:
        body = canonical_json_bytes(payload)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("X-Request-Id", request_id),
            ("X-Should-Retry", "true" if retriable else "false"),
        ]
        if status in {429, 503} and retriable:
            headers.append(("Retry-After", "1"))
        return ApplicationResponse(status, tuple(headers), body)

    def error_response(self, error: AmsError, request_id: str) -> ApplicationResponse:
        status, error_type, api_code, param, message, retriable = _api_error(error)
        return self._json_response(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": param,
                    "code": api_code,
                }
            },
            request_id,
            retriable=retriable,
        )


class OpenAIHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], application: OpenAIApplication):
        self.application = application
        super().__init__(address, OpenAIRequestHandler)


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ams-openai/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self._application().config.socket_timeout_seconds)

    def do_GET(self) -> None:
        self._dispatch(b"")

    def do_POST(self) -> None:
        application = self._application()
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding is not None:
            self._write_response(
                application.error_response(
                    OpenAIProtocolError(
                        "Transfer-Encoding is not supported; send Content-Length",
                        api_code="length_required",
                        http_status=411,
                    ),
                    application._request_id(),
                )
            )
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._write_response(
                application.error_response(
                    OpenAIProtocolError(
                        "a valid Content-Length header is required",
                        api_code="length_required",
                        http_status=411,
                    ),
                    application._request_id(),
                )
            )
            return
        if length > application.config.request_limits.max_body_bytes:
            self.close_connection = True
            self._write_response(
                application.error_response(
                    OpenAIProtocolError(
                        "request body exceeds the configured byte limit",
                        api_code="request_too_large",
                        error_type="request_too_large",
                        http_status=413,
                    ),
                    application._request_id(),
                )
            )
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            self._write_response(
                application.error_response(
                    OpenAIProtocolError(
                        "request body ended before Content-Length bytes arrived",
                        api_code="invalid_json",
                    ),
                    application._request_id(),
                )
            )
            return
        self._dispatch(body)

    def _application(self) -> OpenAIApplication:
        server = self.server
        if not isinstance(server, OpenAIHTTPServer):
            raise RuntimeError("OpenAI request handler is attached to the wrong server")
        return server.application

    def _dispatch(self, body: bytes) -> None:
        headers = {key: value for key, value in self.headers.items()}
        response = self._application().dispatch(self.command, self.path, headers, body)
        self._write_response(response)

    def _write_response(self, response: ApplicationResponse) -> None:
        self.send_response(response.status)
        for name, value in response.headers:
            self.send_header(name, value)
        if isinstance(response.body, ManagedStream):
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if isinstance(response.body, bytes):
            self.wfile.write(response.body)
            return
        try:
            for chunk in response.body:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            response.body.close(cancelled=True)
        else:
            response.body.close()

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_in_thread(
    application: OpenAIApplication,
    host: str = "127.0.0.1",
    port: int = 0,
) -> OpenAIHTTPServer:
    """Construct a localhost server; the caller owns its thread and shutdown."""
    return OpenAIHTTPServer((host, port), application)
