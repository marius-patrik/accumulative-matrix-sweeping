from __future__ import annotations

import http.client
import json
from collections.abc import Iterable
from contextlib import contextmanager
from threading import Event, Thread

from ams.api import (
    GenerationCompleted,
    GenerationEvent,
    GenerationUsage,
    ManagedStream,
    OpenAIApplication,
    OpenAIServerConfig,
    TextDelta,
    serve_in_thread,
)
from ams.errors import AmsError, ErrorCode

MODEL = "ams-glm-4.7-flash"


class ScriptedBackend:
    def __init__(self, events: tuple[GenerationEvent, ...]):
        self.events = events
        self.calls = 0

    def stream(self, request: object, cancellation: Event) -> Iterable[GenerationEvent]:
        self.calls += 1
        for event in self.events:
            if cancellation.is_set():
                return
            yield event


class FailingBackend:
    def stream(self, request: object, cancellation: Event) -> Iterable[GenerationEvent]:
        raise AmsError(
            ErrorCode.BACKEND_FAILURE,
            "fixture backend unavailable",
            retriable=True,
            subsystem="fixture",
        )
        yield  # pragma: no cover


@contextmanager
def running_server(application: OpenAIApplication):
    server = serve_in_thread(application)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(address: tuple[str, int], path: str, payload: bytes):
    connection = http.client.HTTPConnection(*address, timeout=5)
    connection.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-secret"},
    )
    response = connection.getresponse()
    body = response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, headers, body


def test_froq_shaped_responses_sse_round_trips_over_localhost() -> None:
    backend = ScriptedBackend(
        (
            TextDelta("first "),
            TextDelta("line\nsecond line"),
            GenerationCompleted(GenerationUsage(7, 4)),
        )
    )
    application = OpenAIApplication(
        backend,
        OpenAIServerConfig((MODEL,), api_key="local-secret"),
        clock=lambda: 1_234_567_890,
        token_factory=lambda: "fixed",
    )
    request = {
        "model": MODEL,
        "input": [
            {"type": "message", "role": "system", "content": "You are careful."},
            {"type": "message", "role": "user", "content": "Answer."},
        ],
        "reasoning": {"summary": "concise"},
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
    }
    with running_server(application) as address:
        status, headers, body = _post(
            address,
            "/v1/responses",
            json.dumps(request, separators=(",", ":")).encode(),
        )

    assert status == 200
    assert headers["content-type"] == "text/event-stream; charset=utf-8"
    frames = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    assert frames[-1] == b"[DONE]"
    events = [json.loads(frame) for frame in frames[:-1]]
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert (
        "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        == "first line\nsecond line"
    )
    assert events[-1]["response"]["usage"]["total_tokens"] == 11
    assert backend.calls == 1


def test_chat_nonstream_and_malformed_json_have_typed_http_results() -> None:
    backend = ScriptedBackend((TextDelta("hello"), GenerationCompleted(GenerationUsage(2, 1))))
    application = OpenAIApplication(
        backend,
        OpenAIServerConfig((MODEL,), api_key="local-secret"),
        clock=lambda: 10,
        token_factory=lambda: "fixed",
    )
    with running_server(application) as address:
        status, headers, body = _post(
            address,
            "/v1/chat/completions",
            json.dumps(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
            ).encode(),
        )
        bad_status, _, bad_body = _post(
            address,
            "/v1/responses",
            f'{{"model":"{MODEL}","model":"other","input":"hi"}}'.encode(),
        )
        schema_status, _, schema_body = _post(
            address,
            "/v1/responses",
            json.dumps(
                {
                    "model": MODEL,
                    "input": "return json",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "result",
                            "schema": {"type": "object"},
                            "strict": True,
                        }
                    },
                }
            ).encode(),
        )

    result = json.loads(body)
    assert status == 200
    assert headers["x-should-retry"] == "false"
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert bad_status == 400
    assert json.loads(bad_body)["error"]["code"] == "invalid_json"
    assert schema_status == 400
    assert json.loads(schema_body)["error"]["code"] == "unsupported_parameter"
    assert backend.calls == 1


def test_retryable_failure_overload_and_unconsumed_stream_release_are_explicit() -> None:
    failing = OpenAIApplication(
        FailingBackend(),
        OpenAIServerConfig((MODEL,)),
        token_factory=lambda: "failure",
    )
    request = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}).encode()
    failure = failing.dispatch(
        "POST", "/v1/chat/completions", {"Content-Type": "application/json"}, request
    )
    assert failure.status == 500
    failure_headers = dict(failure.headers)
    assert failure_headers["X-Should-Retry"] == "true"
    assert json.loads(failure.body)["error"]["code"] == "backend_failure"

    backend = ScriptedBackend((TextDelta("ok"), GenerationCompleted(GenerationUsage(1, 1))))
    application = OpenAIApplication(
        backend,
        OpenAIServerConfig((MODEL,), max_concurrent_requests=1),
        token_factory=lambda: "slot",
    )
    streaming_request = json.dumps(
        {"model": MODEL, "input": "hi", "stream": True}, separators=(",", ":")
    ).encode()
    first = application.dispatch(
        "POST", "/v1/responses", {"Content-Type": "application/json"}, streaming_request
    )
    assert first.status == 200
    assert isinstance(first.body, ManagedStream)

    overloaded = application.dispatch(
        "POST", "/v1/responses", {"Content-Type": "application/json"}, streaming_request
    )
    assert overloaded.status == 429
    overload_headers = dict(overloaded.headers)
    assert overload_headers["X-Should-Retry"] == "true"
    assert overload_headers["Retry-After"] == "1"

    first.body.close(cancelled=True)
    assert first.body.cancellation.is_set()
    retry = application.dispatch(
        "POST", "/v1/responses", {"Content-Type": "application/json"}, streaming_request
    )
    assert retry.status == 200
    assert isinstance(retry.body, ManagedStream)
    retry.body.close()
