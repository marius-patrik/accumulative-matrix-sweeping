from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_tokenizer import (
    OFFICIAL_GLM4_TOKENIZER_POLICY,
    Glm4AddedToken,
    Glm4TokenizerFile,
    Glm4TokenizerPolicy,
    Glm4TokenizerRuntime,
    _compile_template,
    _load_tokenizer_engine,
    admit_glm4_tokenizer_assets,
)

PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TEMPLATE = """[gMASK]<sop>
{%- if tools -%}{{ tools | tojson(ensure_ascii=False) }}{% endif -%}
{% for m in messages %}<|{{ m.role }}|>{{ m.content }}
{% endfor %}
{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' if enable_thinking else '</think>' -}}
{%- endif -%}
"""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def pinned_file(name: str, payload: bytes) -> Glm4TokenizerFile:
    return Glm4TokenizerFile(name, len(payload), hashlib.sha256(payload).hexdigest())


def fixture_payloads() -> tuple[bytes, bytes, bytes, tuple[Glm4AddedToken, ...]]:
    added = (
        Glm4AddedToken(3, "<eos>", True),
        Glm4AddedToken(4, "<|user|>", True),
        Glm4AddedToken(5, "<|assistant|>", True),
    )
    added_records = [
        {
            "content": token.content,
            "id": token.token_id,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token.special,
        }
        for token in added
    ]
    tokenizer = {
        "added_tokens": added_records,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "byte_fallback": False,
            "continuing_subword_prefix": None,
            "dropout": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "ignore_merges": True,
            "merges": [],
            "type": "BPE",
            "unk_token": None,
            "vocab": {"a": 0, "b": 1, "c": 2},
        },
        "normalizer": None,
        "padding": None,
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": PRETOKENIZER_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "truncation": None,
        "version": "1.0",
    }
    decoder = {
        str(token.token_id): {key: value for key, value in record.items() if key != "id"}
        for token, record in zip(added, added_records, strict=True)
    }
    config = {
        "added_tokens_decoder": decoder,
        "additional_special_tokens": [token.content for token in added],
        "clean_up_tokenization_spaces": False,
        "do_lower_case": False,
        "eos_token": "<eos>",
        "extra_special_tokens": {},
        "model_max_length": 32,
        "pad_token": "<eos>",
        "padding_side": "left",
        "remove_space": False,
        "tokenizer_class": "PreTrainedTokenizer",
    }
    return canonical_json(tokenizer), canonical_json(config), TEMPLATE.encode(), added


def write_fixture(root: Path) -> Glm4TokenizerPolicy:
    tokenizer, config, template, added = fixture_payloads()
    (root / "tokenizer.json").write_bytes(tokenizer)
    (root / "tokenizer_config.json").write_bytes(config)
    (root / "chat_template.jinja").write_bytes(template)
    return Glm4TokenizerPolicy(
        repository="fixture/glm4",
        revision="f" * 40,
        tokenizer_file=pinned_file("tokenizer.json", tokenizer),
        config_file=pinned_file("tokenizer_config.json", config),
        template_file=pinned_file("chat_template.jinja", template),
        base_vocab_size=3,
        merge_count=0,
        model_vocab_size=8,
        model_max_length=32,
        added_tokens=added,
    )


class FakeTokenizer:
    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        assert with_added_tokens
        return 6

    def encode(self, sequence: str, add_special_tokens: bool = True) -> SimpleNamespace:
        assert not add_special_tokens
        return SimpleNamespace(ids=[0, 4] if sequence else [])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return f"{ids}:{skip_special_tokens}"


def test_official_tokenizer_identity_and_unmapped_model_tail_are_pinned() -> None:
    policy = OFFICIAL_GLM4_TOKENIZER_POLICY
    assert policy.revision == "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    assert policy.tokenizer_file.size_bytes == 20_217_442
    assert policy.tokenizer_file.sha256 == (
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
    )
    assert policy.base_vocab_size + len(policy.added_tokens) == 154_856
    assert policy.model_vocab_size == 154_880
    assert policy.added_tokens[8] == Glm4AddedToken(154_828, "<|assistant|>", True)


def test_tokenizer_triplet_is_admitted_with_exact_semantics(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    assets = admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    assert assets.tokenizer_vocab_size == 6
    assert list(assets.unmapped_model_token_ids) == [6, 7]
    assert assets.token_id("<|assistant|>") == 5


def test_digest_drift_fails_before_tokenizer_parse(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    payload = bytearray(path.read_bytes())
    payload[-2] ^= 1
    path.write_bytes(payload)
    with pytest.raises(AmsError) as error:
        admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    assert error.value.code is ErrorCode.INTEGRITY_FAILURE


def test_digest_valid_but_unreviewed_pipeline_is_rejected(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    tokenizer = json.loads(path.read_bytes())
    tokenizer["normalizer"] = {}
    payload = canonical_json(tokenizer)
    path.write_bytes(payload)
    policy = replace(policy, tokenizer_file=pinned_file("tokenizer.json", payload))
    with pytest.raises(AmsError) as error:
        admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    assert error.value.code is ErrorCode.INVALID_PACKAGE


def test_policy_cannot_redirect_asset_paths(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    policy = replace(
        policy,
        tokenizer_file=replace(policy.tokenizer_file, name="../tokenizer.json"),
    )
    with pytest.raises(AmsError) as error:
        admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    assert error.value.code is ErrorCode.INVALID_PACKAGE


def test_unreviewed_tokenizers_runtime_version_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ams.integrations.glm4_tokenizer.importlib.import_module",
        lambda _: SimpleNamespace(__version__="99.0"),
    )
    with pytest.raises(AmsError) as error:
        _load_tokenizer_engine(tmp_path / "tokenizer.json")
    assert error.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_runtime_renders_transformers_compatible_template_and_bounds_tokens(
    tmp_path: Path,
) -> None:
    policy = write_fixture(tmp_path)
    assets = admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    runtime = Glm4TokenizerRuntime(assets, FakeTokenizer(), _compile_template(assets.template))
    rendered = runtime.render_chat(
        [{"role": "user", "content": "héllo"}],
        tools=[{"name": "lookup", "schema": {"type": "object"}}],
        enable_thinking=False,
    )
    assert rendered.startswith('[gMASK]<sop>[{"name": "lookup"')
    assert "<|user|>héllo" in rendered
    assert rendered.endswith("<|assistant|></think>")
    assert runtime.encode(rendered) == (0, 4)
    assert runtime.decode((0, 4)) == "[0, 4]:False"

    with pytest.raises(AmsError) as too_many:
        runtime.encode(rendered, max_tokens=1)
    assert too_many.value.code is ErrorCode.PREFLIGHT_NO_WORKING_SET
    with pytest.raises(AmsError) as unmapped:
        runtime.decode((6,))
    assert unmapped.value.code is ErrorCode.PLAN_INVALID


def test_runtime_rejects_nonfinite_chat_values(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    assets = admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    runtime = Glm4TokenizerRuntime(assets, FakeTokenizer(), _compile_template(assets.template))
    with pytest.raises(AmsError) as error:
        runtime.render_chat([{"role": "user", "content": float("nan")}])
    assert error.value.code is ErrorCode.PLAN_INVALID


def test_runtime_decode_stream_matches_complete_decode(tmp_path: Path) -> None:
    policy = write_fixture(tmp_path)
    assets = admit_glm4_tokenizer_assets(tmp_path, policy=policy)
    runtime = Glm4TokenizerRuntime(
        assets,
        _load_tokenizer_engine(assets.tokenizer_path),
        _compile_template(assets.template),
    )
    stream = runtime.start_decode_stream()
    chunks = [stream.push(token_id) for token_id in (0, 4, 1)]
    suffix = stream.finish()
    assert "".join(chunk or "" for chunk in (*chunks, suffix)) == runtime.decode((0, 4, 1))
