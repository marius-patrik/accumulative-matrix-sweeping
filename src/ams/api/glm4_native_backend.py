"""Model-backed GLM-4 adapter over the persistent native worker protocol."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

from ams.api.contracts import ContentKind, MessageItem, NormalizedOpenAIRequest
from ams.api.generation import (
    GenerationCompleted,
    GenerationEvent,
    GenerationUsage,
    TextDelta,
)
from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_tokenizer import Glm4TokenizerRuntime

_MAX_WORKER_FRAME_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_U64 = (1 << 64) - 1
_READY_FIELDS = {
    "schema_id",
    "evidence",
    "context_capacity_tokens",
    "tokenizer_vocabulary_size",
}
_TOKEN_FIELDS = {"schema_id", "request_id", "index", "token_id"}
_COMPLETED_FIELDS = {"schema_id", "request_id", "output"}
_ERROR_FIELDS = {"schema_id", "request_id", "code", "message"}
_OUTPUT_FIELDS = {
    "schema_id",
    "binding_hash",
    "prompt_tokens",
    "output_token_ids",
    "finish_reason",
    "committed_cache_tokens",
    "generation_steps",
    "cache_heap_bytes",
    "scratch_heap_bytes",
    "scratch_logical_bytes",
}
_RETRIABLE_CODES = {
    ErrorCode.PREFLIGHT_NO_BACKING,
    ErrorCode.PREFLIGHT_NO_WORKING_SET,
    ErrorCode.RESERVATION_LOST,
    ErrorCode.BROKER_VIOLATION,
    ErrorCode.IO_FAILURE,
    ErrorCode.BACKEND_FAILURE,
    ErrorCode.TRANSACTION_FAILURE,
    ErrorCode.DEADLINE_EXCEEDED,
}


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


def _backend_error(
    code: ErrorCode,
    message: str,
    *,
    retriable: bool = False,
    phase: str = "decode",
) -> AmsError:
    return AmsError(
        code,
        message,
        retriable=retriable,
        phase=phase,
        subsystem="glm4-native-backend",
    )


def _parse_worker_frame(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
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
        raise _backend_error(
            ErrorCode.BACKEND_FAILURE,
            "native worker emitted malformed strict JSON",
        ) from exc
    if not isinstance(value, dict):
        raise _backend_error(
            ErrorCode.BACKEND_FAILURE,
            "native worker frame is not a JSON object",
        )
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise _backend_error(
            ErrorCode.BACKEND_FAILURE,
            f"native worker {description} fields changed",
        )


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _backend_error(
            ErrorCode.BACKEND_FAILURE,
            f"native worker {description} is invalid",
        )
    return value


def _regular_absolute_file(path: Path, description: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise _backend_error(
            ErrorCode.PLAN_INVALID,
            f"{description} must be an absolute nonsymlink regular file",
            phase="preflight",
        )
    return path


@dataclass(frozen=True, slots=True)
class Glm4NativeBackendConfig:
    """Exact local native-worker identity and bounded process controls."""

    native_binary: Path
    binding_envelope: Path
    expected_binding_hash: str
    verification_buffer_bytes: int = 1024 * 1024
    default_max_output_tokens: int = 1024
    startup_timeout_seconds: float = 60.0
    frame_poll_seconds: float = 0.05
    cancellation_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "native_binary",
            _regular_absolute_file(Path(self.native_binary), "native binary"),
        )
        object.__setattr__(
            self,
            "binding_envelope",
            _regular_absolute_file(Path(self.binding_envelope), "binding envelope"),
        )
        digest = self.expected_binding_hash
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise _backend_error(
                ErrorCode.PLAN_INVALID,
                "expected native binding hash is invalid",
                phase="preflight",
            )
        for name in ("verification_buffer_bytes", "default_max_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _backend_error(
                    ErrorCode.PLAN_INVALID,
                    f"{name} must be a positive integer",
                    phase="preflight",
                )
        if self.verification_buffer_bytes > 64 * 1024 * 1024:
            raise _backend_error(
                ErrorCode.PLAN_INVALID,
                "verification buffer exceeds the native admission limit",
                phase="preflight",
            )
        for name in (
            "startup_timeout_seconds",
            "frame_poll_seconds",
            "cancellation_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise _backend_error(
                    ErrorCode.PLAN_INVALID,
                    f"{name} must be a positive finite number",
                    phase="preflight",
                )


class _NativeWorker:
    def __init__(
        self,
        config: Glm4NativeBackendConfig,
        tokenizer_vocabulary_size: int,
    ) -> None:
        self.config = config
        self._frames: Queue[bytes | AmsError | None] = Queue(maxsize=4)
        self._stderr = bytearray()
        self._stderr_lock = Lock()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                [
                    str(config.native_binary),
                    "worker",
                    str(config.binding_envelope),
                    str(config.verification_buffer_bytes),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker process could not start",
                retriable=True,
                phase="startup",
            ) from exc
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.terminate()
            raise _backend_error(
                ErrorCode.INTERNAL_INVARIANT,
                "native worker process pipes were not created",
                phase="startup",
            )
        self._stdout_thread = Thread(
            target=self._pump_stdout,
            name="ams-native-worker-stdout",
            daemon=True,
        )
        self._stderr_thread = Thread(
            target=self._pump_stderr,
            name="ams-native-worker-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._wait_ready()
            self.context_capacity_tokens = _integer(
                ready["context_capacity_tokens"],
                "context capacity",
                minimum=1,
            )
            self.tokenizer_vocabulary_size = _integer(
                ready["tokenizer_vocabulary_size"],
                "tokenizer vocabulary size",
                minimum=1,
            )
            if self.tokenizer_vocabulary_size != tokenizer_vocabulary_size:
                raise _backend_error(
                    ErrorCode.CAPABILITY_MISMATCH,
                    "native worker and tokenizer vocabulary sizes differ",
                    phase="startup",
                )
        except Exception:
            self.terminate()
            raise

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def _pump_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline(_MAX_WORKER_FRAME_BYTES + 2)
                if not line:
                    break
                if len(line) > _MAX_WORKER_FRAME_BYTES + 1 or not line.endswith(b"\n"):
                    self._frames.put(
                        _backend_error(
                            ErrorCode.BACKEND_FAILURE,
                            "native worker output frame exceeds the admitted byte bound",
                        )
                    )
                    return
                self._frames.put(line[:-1])
        except OSError as exc:
            error = _backend_error(
                ErrorCode.IO_FAILURE,
                "native worker output pipe failed",
                retriable=True,
            )
            error.__cause__ = exc
            self._frames.put(error)
        finally:
            self._frames.put(None)

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while chunk := self.process.stderr.read(4096):
                with self._stderr_lock:
                    remaining = _MAX_STDERR_BYTES - len(self._stderr)
                    if remaining > 0:
                        self._stderr.extend(chunk[:remaining])
        except OSError:
            return

    def _wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _backend_error(
                    ErrorCode.DEADLINE_EXCEEDED,
                    "native worker did not become ready before the startup deadline",
                    retriable=True,
                    phase="startup",
                )
            frame = self.poll_frame(min(remaining, self.config.frame_poll_seconds))
            if frame is None:
                continue
            _exact_fields(frame, _READY_FIELDS, "ready frame")
            if frame["schema_id"] != "ams.native.worker.ready.v1":
                raise _backend_error(
                    ErrorCode.CAPABILITY_MISMATCH,
                    "native worker ready schema is unsupported",
                    phase="startup",
                )
            evidence = frame["evidence"]
            if not isinstance(evidence, dict):
                raise _backend_error(
                    ErrorCode.BACKEND_FAILURE,
                    "native worker ready evidence is invalid",
                    phase="startup",
                )
            if evidence.get("binding_hash") != self.config.expected_binding_hash:
                raise _backend_error(
                    ErrorCode.INTEGRITY_FAILURE,
                    "native worker admitted an unexpected binding identity",
                    phase="startup",
                )
            return frame

    def poll_frame(self, timeout: float) -> dict[str, Any] | None:
        try:
            item = self._frames.get(timeout=timeout)
        except Empty:
            if self.process.poll() is not None:
                raise _backend_error(
                    ErrorCode.BACKEND_FAILURE,
                    "native worker exited without a terminal frame",
                    retriable=True,
                ) from None
            return None
        if item is None:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker closed its output stream",
                retriable=True,
            )
        if isinstance(item, AmsError):
            raise item
        return _parse_worker_frame(item)

    def send(self, frame: dict[str, Any]) -> None:
        payload = canonical_json_bytes(frame) + b"\n"
        if len(payload) - 1 > _MAX_WORKER_FRAME_BYTES:
            raise _backend_error(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "native worker request frame exceeds the admitted byte bound",
                phase="preflight",
            )
        if not self.alive or self.process.stdin is None:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker is not running",
                retriable=True,
            )
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise _backend_error(
                ErrorCode.IO_FAILURE,
                "native worker input pipe failed",
                retriable=True,
            ) from exc

    def shutdown(self) -> None:
        if not self.alive:
            self.terminate()
            return
        try:
            self.send({"schema_id": "ams.native.worker.shutdown.v1"})
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=self.config.cancellation_timeout_seconds)
        except (AmsError, OSError, subprocess.TimeoutExpired):
            self.terminate()
        else:
            self._close_pipes()

    def terminate(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None:
                with suppress(OSError):
                    pipe.close()


class Glm4NativeBackend:
    """Text-only greedy OpenAI backend over one restartable persistent native worker."""

    def __init__(
        self,
        tokenizer: Glm4TokenizerRuntime,
        config: Glm4NativeBackendConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self._request_lock = Lock()
        self._lifecycle_lock = Lock()
        self._request_id = 0
        self._worker: _NativeWorker | None = self._start_worker()
        if config.default_max_output_tokens > self._worker.context_capacity_tokens:
            self.close()
            raise _backend_error(
                ErrorCode.PLAN_INVALID,
                "default output limit exceeds the native context capacity",
                phase="preflight",
            )

    def __enter__(self) -> Glm4NativeBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _start_worker(self) -> _NativeWorker:
        return _NativeWorker(self.config, self.tokenizer.assets.tokenizer_vocab_size)

    def _ensure_worker(self) -> _NativeWorker:
        with self._lifecycle_lock:
            if self._worker is not None and self._worker.alive:
                return self._worker
            if self._worker is not None:
                self._worker.terminate()
            self._worker = self._start_worker()
            return self._worker

    def _discard_worker(self, worker: _NativeWorker) -> None:
        with self._lifecycle_lock:
            if self._worker is worker:
                worker.terminate()
                self._worker = None

    def close(self) -> None:
        with self._request_lock:
            with self._lifecycle_lock:
                worker = self._worker
                self._worker = None
            if worker is not None:
                worker.shutdown()

    def _next_request_id(self) -> int:
        self._request_id = 1 if self._request_id >= _MAX_U64 else self._request_id + 1
        return self._request_id

    @staticmethod
    def _unsupported(message: str) -> AmsError:
        return _backend_error(
            ErrorCode.UNSUPPORTED_OP,
            message,
            phase="preflight",
        )

    @staticmethod
    def _text(item: MessageItem) -> str:
        if any(part.kind is not ContentKind.TEXT for part in item.content):
            raise Glm4NativeBackend._unsupported(
                "native GLM-4 backend currently accepts text input only"
            )
        return "".join(part.text or "" for part in item.content)

    def _prepare_request(
        self,
        request: NormalizedOpenAIRequest,
        worker: _NativeWorker,
    ) -> tuple[tuple[int, ...], int]:
        if request.tools or request.tool_choice is not None:
            raise self._unsupported("native GLM-4 backend tool calling is not qualified yet")
        if request.structured_output is not None:
            raise self._unsupported("native GLM-4 backend structured output is not qualified yet")
        if request.temperature not in {None, 0.0} or request.top_p not in {None, 1.0}:
            raise self._unsupported("native GLM-4 backend currently supports greedy decoding only")
        if request.reasoning_effort not in {None, "none"} or request.reasoning_summary is not None:
            raise self._unsupported("native GLM-4 backend reasoning output is not qualified yet")
        if request.prompt_cache_key is not None:
            raise self._unsupported("native GLM-4 backend prompt caching is not qualified yet")
        messages: list[dict[str, str]] = []
        for item in request.input_items:
            if not isinstance(item, MessageItem):
                raise self._unsupported(
                    "native GLM-4 backend currently accepts message history only"
                )
            role = "system" if item.role == "developer" else item.role
            messages.append({"role": role, "content": self._text(item)})
        max_new_tokens = request.max_output_tokens or self.config.default_max_output_tokens
        maximum_prompt_tokens = worker.context_capacity_tokens - max_new_tokens + 1
        if maximum_prompt_tokens <= 0:
            raise _backend_error(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "requested output leaves no native prompt capacity",
                phase="preflight",
            )
        prompt = self.tokenizer.encode_chat(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            clear_thinking=True,
            max_tokens=maximum_prompt_tokens,
        )
        if not prompt:
            raise _backend_error(
                ErrorCode.PLAN_INVALID,
                "GLM chat template produced an empty prompt",
                phase="preflight",
            )
        return prompt, max_new_tokens

    @staticmethod
    def _worker_error(frame: dict[str, Any], request_id: int) -> AmsError:
        _exact_fields(frame, _ERROR_FIELDS, "error frame")
        if frame["schema_id"] != "ams.native.worker.error.v1":
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker terminal error schema is unsupported",
            )
        if _integer(frame["request_id"], "error request ID") != request_id:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker error request ID changed",
            )
        message = frame["message"]
        if not isinstance(message, str) or not message:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker error message is invalid",
            )
        try:
            code = ErrorCode(frame["code"])
        except (TypeError, ValueError) as exc:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker error code is unknown",
            ) from exc
        return _backend_error(code, message, retriable=code in _RETRIABLE_CODES)

    def _validate_completed(
        self,
        frame: dict[str, Any],
        request_id: int,
        prompt_tokens: int,
        max_new_tokens: int,
        output_token_ids: list[int],
    ) -> str:
        _exact_fields(frame, _COMPLETED_FIELDS, "completed frame")
        if (
            frame["schema_id"] != "ams.native.worker.completed.v1"
            or _integer(frame["request_id"], "completion request ID") != request_id
        ):
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker completion identity changed",
            )
        output = frame["output"]
        if not isinstance(output, dict):
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker completion output is invalid",
            )
        _exact_fields(output, _OUTPUT_FIELDS, "generation output")
        raw_output_token_ids = output["output_token_ids"]
        if not isinstance(raw_output_token_ids, list):
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker output-token inventory is invalid",
            )
        validated_output_token_ids = [
            _integer(token_id, "completed output token ID") for token_id in raw_output_token_ids
        ]
        if (
            output["schema_id"] != "ams.native.glm4-greedy-output.v1"
            or output["binding_hash"] != self.config.expected_binding_hash
            or _integer(output["prompt_tokens"], "prompt token count") != prompt_tokens
            or validated_output_token_ids != output_token_ids
            or len(output_token_ids) > max_new_tokens
        ):
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker completion contradicted the observed request",
            )
        for field in (
            "committed_cache_tokens",
            "generation_steps",
            "cache_heap_bytes",
            "scratch_heap_bytes",
            "scratch_logical_bytes",
        ):
            _integer(output[field], field)
        finish_reason = output["finish_reason"]
        if finish_reason not in {"end_of_sequence", "length"}:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker finish reason is unsupported",
            )
        return "stop" if finish_reason == "end_of_sequence" else "length"

    @staticmethod
    def _validate_token(
        frame: dict[str, Any],
        request_id: int,
        expected_index: int,
        vocabulary_size: int,
    ) -> int:
        _exact_fields(frame, _TOKEN_FIELDS, "token frame")
        if (
            frame["schema_id"] != "ams.native.worker.token.v1"
            or _integer(frame["request_id"], "token request ID") != request_id
            or _integer(frame["index"], "token index") != expected_index
        ):
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker token ordering changed",
            )
        token_id = _integer(frame["token_id"], "token ID")
        if token_id >= vocabulary_size:
            raise _backend_error(
                ErrorCode.BACKEND_FAILURE,
                "native worker emitted an unmapped tokenizer ID",
            )
        return token_id

    @staticmethod
    def _send_cancel(worker: _NativeWorker, request_id: int) -> None:
        worker.send(
            {
                "schema_id": "ams.native.worker.cancel.v1",
                "request_id": request_id,
            }
        )

    def _cancel_and_drain(self, worker: _NativeWorker, request_id: int) -> None:
        try:
            self._send_cancel(worker, request_id)
            deadline = time.monotonic() + self.config.cancellation_timeout_seconds
            while time.monotonic() < deadline:
                frame = worker.poll_frame(self.config.frame_poll_seconds)
                if frame is None:
                    continue
                if _integer(frame.get("request_id"), "cancellation request ID") != request_id:
                    break
                if frame.get("schema_id") in {
                    "ams.native.worker.completed.v1",
                    "ams.native.worker.error.v1",
                }:
                    return
        except AmsError:
            pass
        self._discard_worker(worker)

    def stream(
        self,
        request: NormalizedOpenAIRequest,
        cancellation: Event,
    ) -> Iterable[GenerationEvent]:
        if not self._request_lock.acquire(blocking=False):
            raise _backend_error(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "native GLM-4 backend already has an active request",
                retriable=True,
                phase="preflight",
            )
        worker: _NativeWorker | None = None
        request_id: int | None = None
        terminal = False
        try:
            if cancellation.is_set():
                raise _backend_error(ErrorCode.CANCELLED, "request was cancelled")
            worker = self._ensure_worker()
            prompt, max_new_tokens = self._prepare_request(request, worker)
            request_id = self._next_request_id()
            worker.send(
                {
                    "schema_id": "ams.native.worker.generate.v1",
                    "request_id": request_id,
                    "prompt_token_ids": list(prompt),
                    "max_new_tokens": max_new_tokens,
                }
            )
            decoder = self.tokenizer.start_decode_stream(skip_special_tokens=False)
            output_token_ids: list[int] = []
            cancel_sent = False
            while True:
                if cancellation.is_set() and not cancel_sent:
                    self._send_cancel(worker, request_id)
                    cancel_sent = True
                frame = worker.poll_frame(self.config.frame_poll_seconds)
                if frame is None:
                    continue
                schema = frame.get("schema_id")
                if schema == "ams.native.worker.token.v1":
                    token_id = self._validate_token(
                        frame,
                        request_id,
                        len(output_token_ids),
                        worker.tokenizer_vocabulary_size,
                    )
                    output_token_ids.append(token_id)
                    if cancellation.is_set():
                        continue
                    chunk = decoder.push(token_id)
                    if chunk is not None:
                        yield TextDelta(chunk)
                    continue
                if schema == "ams.native.worker.error.v1":
                    terminal = True
                    raise self._worker_error(frame, request_id)
                if schema != "ams.native.worker.completed.v1":
                    raise _backend_error(
                        ErrorCode.BACKEND_FAILURE,
                        "native worker emitted an unexpected frame",
                    )
                finish_reason = self._validate_completed(
                    frame,
                    request_id,
                    len(prompt),
                    max_new_tokens,
                    output_token_ids,
                )
                terminal = True
                if cancellation.is_set():
                    raise _backend_error(ErrorCode.CANCELLED, "request was cancelled")
                suffix = decoder.finish()
                if suffix is not None:
                    yield TextDelta(suffix)
                yield GenerationCompleted(
                    GenerationUsage(len(prompt), len(output_token_ids)),
                    finish_reason,
                )
                return
        finally:
            if worker is not None and request_id is not None and not terminal:
                self._cancel_and_drain(worker, request_id)
            self._request_lock.release()
