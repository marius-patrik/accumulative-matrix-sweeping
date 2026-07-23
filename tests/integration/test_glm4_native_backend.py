from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event

import pytest

from ams.api import (
    GenerationCompleted,
    Glm4NativeBackend,
    Glm4NativeBackendConfig,
    ManagedStream,
    OpenAIApplication,
    OpenAIServerConfig,
    TextDelta,
    normalize_chat_completions_request,
)
from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_tokenizer import (
    Glm4TokenizerRuntime,
    _compile_template,
    _load_tokenizer_engine,
    admit_glm4_tokenizer_assets,
)
from ams.ops import GlmPackageWeights, serialize_glm4_native_binding_plan
from tests.integration.test_glm4_tokenizer_contract import write_fixture
from tests.invariant.test_mini_glm4_forward import build_mini_glm4_package

MODEL = "ams-glm-4.7-mini-native"
EXPECTED_TEXT = "<|assistant|><|assistant|>"


def _request(*, stream: bool, temperature: float = 0.0) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "a"}],
        "max_tokens": 2,
        "temperature": temperature,
        "stream": stream,
    }


def _responses_request(*, stream: bool) -> dict[str, object]:
    return {
        "model": MODEL,
        "input": "a",
        "max_output_tokens": 2,
        "temperature": 0,
        "stream": stream,
    }


def test_native_backend_serves_tokenized_model_output_and_recovers_after_disconnect(
    tmp_path: Path,
) -> None:
    binary = os.environ.get("AMS_NATIVE_BINARY")
    if binary is None:
        pytest.skip("model-backed native adapter is exercised by the post-build verification step")

    package_source = tmp_path / "package-source"
    package_source.mkdir()
    package_root, _, _ = build_mini_glm4_package(package_source)
    package = GlmPackageWeights.open(package_root, linear_arena_bytes=64)

    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    policy = write_fixture(tokenizer_root)
    assets = admit_glm4_tokenizer_assets(tokenizer_root, policy=policy)
    tokenizer = Glm4TokenizerRuntime(
        assets,
        _load_tokenizer_engine(assets.tokenizer_path),
        _compile_template(assets.template),
    )

    plan = package.native_glm4_binding_plan(
        context_capacity_tokens=16,
        tokenizer_vocabulary_size=assets.tokenizer_vocab_size,
        eos_token_ids=(assets.token_id("<eos>"),),
    )
    envelope = tmp_path / "native-binding.json"
    envelope.write_bytes(serialize_glm4_native_binding_plan(plan))
    config = Glm4NativeBackendConfig(
        Path(binary),
        envelope,
        plan.binding_hash,
        verification_buffer_bytes=64,
        default_max_output_tokens=2,
    )

    with Glm4NativeBackend(tokenizer, config) as backend:
        normalized = normalize_chat_completions_request(_request(stream=False))
        events = tuple(backend.stream(normalized, Event()))
        assert "".join(event.text for event in events if isinstance(event, TextDelta)) == (
            EXPECTED_TEXT
        )
        completed = [event for event in events if isinstance(event, GenerationCompleted)]
        assert len(completed) == 1
        assert completed[0].finish_reason == "length"
        assert completed[0].usage.input_tokens == 3
        assert completed[0].usage.output_tokens == 2

        assert backend._worker is not None
        original_pid = backend._worker.process.pid
        backend._worker.process.kill()
        backend._worker.process.wait(timeout=5)
        restarted = tuple(backend.stream(normalized, Event()))
        assert "".join(event.text for event in restarted if isinstance(event, TextDelta)) == (
            EXPECTED_TEXT
        )
        assert backend._worker is not None
        assert backend._worker.process.pid != original_pid

        unsupported = normalize_chat_completions_request(_request(stream=False, temperature=1.0))
        with pytest.raises(AmsError) as error:
            tuple(backend.stream(unsupported, Event()))
        assert error.value.code is ErrorCode.UNSUPPORTED_OP

        application = OpenAIApplication(
            backend,
            OpenAIServerConfig((MODEL,)),
            clock=lambda: 10,
            token_factory=lambda: "native",
        )
        streaming = application.dispatch(
            "POST",
            "/v1/responses",
            {"Content-Type": "application/json"},
            canonical_json_bytes(_responses_request(stream=True)),
        )
        assert streaming.status == 200
        assert isinstance(streaming.body, ManagedStream)
        while True:
            chunk = next(streaming.body)
            if b'"type":"response.output_text.delta"' in chunk:
                break
        streaming.body.close(cancelled=True)
        assert streaming.body.cancellation.is_set()

        retried = application.dispatch(
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            canonical_json_bytes(_request(stream=False)),
        )
        assert retried.status == 200
        assert isinstance(retried.body, bytes)
        response = json.loads(retried.body)
        assert response["choices"][0]["message"]["content"] == EXPECTED_TEXT
        assert response["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
