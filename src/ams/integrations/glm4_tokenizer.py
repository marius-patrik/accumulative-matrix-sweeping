"""Pinned GLM-4 tokenizer admission and bounded execution."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ams.errors import AmsError, ErrorCode

_TOKENIZER_TOP_LEVEL = {
    "added_tokens",
    "decoder",
    "model",
    "normalizer",
    "padding",
    "post_processor",
    "pre_tokenizer",
    "truncation",
    "version",
}
_TOKENIZER_MODEL_FIELDS = {
    "byte_fallback",
    "continuing_subword_prefix",
    "dropout",
    "end_of_word_suffix",
    "fuse_unk",
    "ignore_merges",
    "merges",
    "type",
    "unk_token",
    "vocab",
}
_TOKENIZER_CONFIG_FIELDS = {
    "added_tokens_decoder",
    "additional_special_tokens",
    "clean_up_tokenization_spaces",
    "do_lower_case",
    "eos_token",
    "extra_special_tokens",
    "model_max_length",
    "pad_token",
    "padding_side",
    "remove_space",
    "tokenizer_class",
}
_ADDED_TOKEN_FIELDS = {
    "content",
    "id",
    "lstrip",
    "normalized",
    "rstrip",
    "single_word",
    "special",
}
_GLM4_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_GLM4_PRETOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": _GLM4_PRETOKENIZER_PATTERN},
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
}
_GLM4_POST_PROCESSOR = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": False,
    "use_regex": True,
}
_GLM4_DECODER = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": True,
    "use_regex": True,
}
_HASH_READ_BYTES = 1024 * 1024
_MAX_CHAT_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_CHAT_MESSAGES = 4096
_MAX_CHAT_TOOLS = 256
_TOKENIZERS_VERSION = "0.22.2"
_JINJA2_VERSION = "3.1.6"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Glm4TokenizerFile:
    """Immutable identity for one tokenizer asset."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Glm4AddedToken:
    """One exact added-token record."""

    token_id: int
    content: str
    special: bool


@dataclass(frozen=True)
class Glm4TokenizerPolicy:
    """Reviewed tokenizer identity and semantic contract."""

    repository: str
    revision: str
    tokenizer_file: Glm4TokenizerFile
    config_file: Glm4TokenizerFile
    template_file: Glm4TokenizerFile
    base_vocab_size: int
    merge_count: int
    model_vocab_size: int
    model_max_length: int
    added_tokens: tuple[Glm4AddedToken, ...]


_OFFICIAL_ADDED_TOKENS = tuple(
    Glm4AddedToken(154820 + offset, content, offset < 18)
    for offset, content in enumerate(
        (
            "<|endoftext|>",
            "[MASK]",
            "[gMASK]",
            "[sMASK]",
            "<sop>",
            "<eop>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "<|begin_of_image|>",
            "<|end_of_image|>",
            "<|begin_of_video|>",
            "<|end_of_video|>",
            "<|begin_of_audio|>",
            "<|end_of_audio|>",
            "<|begin_of_transcription|>",
            "<|end_of_transcription|>",
            "<|code_prefix|>",
            "<|code_middle|>",
            "<|code_suffix|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
            "/nothink",
            "<|begin_of_box|>",
            "<|end_of_box|>",
            "<|image|>",
            "<|video|>",
        )
    )
)

OFFICIAL_GLM4_TOKENIZER_POLICY = Glm4TokenizerPolicy(
    repository="zai-org/GLM-4.7-Flash",
    revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    tokenizer_file=Glm4TokenizerFile(
        "tokenizer.json",
        20_217_442,
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    config_file=Glm4TokenizerFile(
        "tokenizer_config.json",
        7_226,
        "31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d",
    ),
    template_file=Glm4TokenizerFile(
        "chat_template.jinja",
        3_120,
        "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375",
    ),
    base_vocab_size=154_820,
    merge_count=321_649,
    model_vocab_size=154_880,
    model_max_length=128_000,
    added_tokens=_OFFICIAL_ADDED_TOKENS,
)


@dataclass(frozen=True)
class Glm4TokenizerAssets:
    """Admitted asset paths and the semantics derived from them."""

    root: Path
    tokenizer_path: Path
    config_path: Path
    template_path: Path
    template: str
    policy: Glm4TokenizerPolicy

    @property
    def tokenizer_vocab_size(self) -> int:
        return self.policy.base_vocab_size + len(self.policy.added_tokens)

    @property
    def unmapped_model_token_ids(self) -> range:
        return range(self.tokenizer_vocab_size, self.policy.model_vocab_size)

    def token_id(self, content: str) -> int:
        for token in self.policy.added_tokens:
            if token.content == content:
                return token.token_id
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "requested token is absent from the admitted GLM-4 tokenizer",
            subsystem="glm4_tokenizer",
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"{name} is not strict UTF-8 JSON",
            subsystem="glm4_tokenizer",
        ) from exc
    if not isinstance(value, dict):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"{name} must contain a JSON object",
            subsystem="glm4_tokenizer",
        )
    return value


def _read_pinned_file(root: Path, expected: Glm4TokenizerFile) -> tuple[Path, bytes]:
    path = root / expected.name
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"pinned tokenizer asset is missing or indirect: {expected.name}",
                subsystem="glm4_tokenizer",
            )
        size = path.stat().st_size
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            f"cannot inspect pinned tokenizer asset: {expected.name}",
            subsystem="glm4_tokenizer",
        ) from exc
    if size != expected.size_bytes:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            f"pinned tokenizer asset size changed: {expected.name}",
            subsystem="glm4_tokenizer",
            evidence={"actual_bytes": size, "expected_bytes": expected.size_bytes},
        )

    digest = hashlib.sha256()
    payload = bytearray()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_HASH_READ_BYTES):
                digest.update(chunk)
                payload.extend(chunk)
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            f"cannot read pinned tokenizer asset: {expected.name}",
            subsystem="glm4_tokenizer",
        ) from exc
    if digest.hexdigest() != expected.sha256:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            f"pinned tokenizer asset digest changed: {expected.name}",
            subsystem="glm4_tokenizer",
        )
    return path, bytes(payload)


def _validate_policy(policy: Glm4TokenizerPolicy) -> None:
    files = (policy.tokenizer_file, policy.config_file, policy.template_file)
    if [item.name for item in files] != [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4 tokenizer policy filenames are invalid",
            subsystem="glm4_tokenizer",
        )
    if any(
        item.size_bytes <= 0
        or item.size_bytes > 32 * 1024 * 1024
        or _SHA256.fullmatch(item.sha256) is None
        for item in files
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4 tokenizer policy file identity is invalid",
            subsystem="glm4_tokenizer",
        )
    if (
        not policy.repository
        or re.fullmatch(r"[0-9a-f]{40}", policy.revision) is None
        or policy.base_vocab_size <= 0
        or policy.merge_count < 0
        or policy.model_vocab_size <= policy.base_vocab_size
        or policy.model_max_length <= 0
        or not policy.added_tokens
        or len({token.content for token in policy.added_tokens}) != len(policy.added_tokens)
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4 tokenizer policy semantics are invalid",
            subsystem="glm4_tokenizer",
        )


def _expected_added_token(token: Glm4AddedToken) -> dict[str, Any]:
    return {
        "content": token.content,
        "id": token.token_id,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": token.special,
    }


def _expected_decoder_token(token: Glm4AddedToken) -> dict[str, Any]:
    value = _expected_added_token(token)
    del value["id"]
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    if set(value) != fields:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"{name} fields are missing or unreviewed",
            subsystem="glm4_tokenizer",
        )


def _validate_tokenizer(value: dict[str, Any], policy: Glm4TokenizerPolicy) -> None:
    _require_exact_fields(value, _TOKENIZER_TOP_LEVEL, name="tokenizer")
    if (
        value["version"] != "1.0"
        or value["truncation"] is not None
        or value["padding"] is not None
        or value["normalizer"] is not None
        or value["pre_tokenizer"] != _GLM4_PRETOKENIZER
        or value["post_processor"] != _GLM4_POST_PROCESSOR
        or value["decoder"] != _GLM4_DECODER
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer pipeline does not match reviewed GLM-4 semantics",
            subsystem="glm4_tokenizer",
        )

    model = value["model"]
    if not isinstance(model, dict):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer model is invalid",
            subsystem="glm4_tokenizer",
        )
    _require_exact_fields(model, _TOKENIZER_MODEL_FIELDS, name="tokenizer model")
    fixed_model = {
        "byte_fallback": False,
        "continuing_subword_prefix": None,
        "dropout": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "ignore_merges": True,
        "type": "BPE",
        "unk_token": None,
    }
    if any(model[name] != expected for name, expected in fixed_model.items()):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer BPE semantics are unsupported",
            subsystem="glm4_tokenizer",
        )

    vocab = model["vocab"]
    if (
        not isinstance(vocab, dict)
        or len(vocab) != policy.base_vocab_size
        or any(not isinstance(key, str) for key in vocab)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in vocab.values())
        or set(vocab.values()) != set(range(policy.base_vocab_size))
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer base vocabulary is not contiguous",
            subsystem="glm4_tokenizer",
        )
    merges = model["merges"]
    if (
        not isinstance(merges, list)
        or len(merges) != policy.merge_count
        or any(
            not isinstance(merge, list)
            or len(merge) != 2
            or any(not isinstance(piece, str) for piece in merge)
            for merge in merges
        )
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer merge inventory is invalid",
            subsystem="glm4_tokenizer",
        )

    expected_added = [_expected_added_token(token) for token in policy.added_tokens]
    if value["added_tokens"] != expected_added:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer added-token inventory changed",
            subsystem="glm4_tokenizer",
        )
    expected_ids = list(
        range(policy.base_vocab_size, policy.base_vocab_size + len(policy.added_tokens))
    )
    if [token.token_id for token in policy.added_tokens] != expected_ids:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "reviewed added-token policy is not contiguous",
            subsystem="glm4_tokenizer",
        )
    if policy.model_vocab_size < policy.base_vocab_size + len(policy.added_tokens):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "model vocabulary cannot contain the reviewed tokenizer",
            subsystem="glm4_tokenizer",
        )


def _validate_config(value: dict[str, Any], policy: Glm4TokenizerPolicy) -> None:
    _require_exact_fields(value, _TOKENIZER_CONFIG_FIELDS, name="tokenizer config")
    expected_decoder = {
        str(token.token_id): _expected_decoder_token(token) for token in policy.added_tokens
    }
    special_tokens = [token.content for token in policy.added_tokens if token.special]
    eos = policy.added_tokens[0].content
    if (
        value["added_tokens_decoder"] != expected_decoder
        or value["additional_special_tokens"] != special_tokens
        or value["clean_up_tokenization_spaces"] is not False
        or value["do_lower_case"] is not False
        or value["eos_token"] != eos
        or value["extra_special_tokens"] != {}
        or value["model_max_length"] != policy.model_max_length
        or value["pad_token"] != eos
        or value["padding_side"] != "left"
        or value["remove_space"] is not False
        or value["tokenizer_class"] != "PreTrainedTokenizer"
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "tokenizer configuration does not match reviewed GLM-4 semantics",
            subsystem="glm4_tokenizer",
        )


def admit_glm4_tokenizer_assets(
    root: Path,
    *,
    policy: Glm4TokenizerPolicy = OFFICIAL_GLM4_TOKENIZER_POLICY,
) -> Glm4TokenizerAssets:
    """Admit only a complete tokenizer triplet with reviewed identity and semantics."""

    root = Path(root)
    _validate_policy(policy)
    if root.is_symlink() or not root.is_dir():
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "GLM-4 tokenizer root is missing or indirect",
            subsystem="glm4_tokenizer",
        )
    tokenizer_path, tokenizer_payload = _read_pinned_file(root, policy.tokenizer_file)
    config_path, config_payload = _read_pinned_file(root, policy.config_file)
    template_path, template_payload = _read_pinned_file(root, policy.template_file)
    _validate_tokenizer(_read_json(tokenizer_payload, name="tokenizer.json"), policy)
    _validate_config(_read_json(config_payload, name="tokenizer_config.json"), policy)
    try:
        template = template_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "chat template is not UTF-8",
            subsystem="glm4_tokenizer",
        ) from exc
    if (
        not template.startswith("[gMASK]<sop>\n")
        or "{%- if add_generation_prompt -%}" not in template
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "chat template does not expose reviewed GLM-4 framing",
            subsystem="glm4_tokenizer",
        )
    return Glm4TokenizerAssets(
        root=root,
        tokenizer_path=tokenizer_path,
        config_path=config_path,
        template_path=template_path,
        template=template,
        policy=policy,
    )


class _Encoding(Protocol):
    ids: list[int]


class _TokenizerEngine(Protocol):
    def encode(self, sequence: str, add_special_tokens: bool = True) -> _Encoding: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...


class _StreamingDecoder(Protocol):
    def step(self, tokenizer: _TokenizerEngine, token_id: int) -> str | None: ...


class Glm4TokenizerDecodeStream:
    """Exact incremental decode with a final full-decode consistency check."""

    def __init__(
        self,
        runtime: Glm4TokenizerRuntime,
        decoder: _StreamingDecoder,
        *,
        skip_special_tokens: bool,
    ) -> None:
        self._runtime = runtime
        self._decoder = decoder
        self._skip_special_tokens = skip_special_tokens
        self._token_ids: list[int] = []
        self._chunks: list[str] = []
        self._finished = False

    def push(self, token_id: int) -> str | None:
        if self._finished:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "cannot append to a finished GLM decode stream",
                subsystem="glm4_tokenizer",
            )
        self._runtime._validate_decode_token(token_id)
        try:
            chunk = self._decoder.step(self._runtime._engine, token_id)
        except Exception as exc:
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers failed to stream-decode GLM output",
                subsystem="glm4_tokenizer",
            ) from exc
        if chunk is not None and (not isinstance(chunk, str) or not chunk):
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers returned an invalid GLM decode chunk",
                subsystem="glm4_tokenizer",
            )
        self._token_ids.append(token_id)
        if chunk is not None:
            self._chunks.append(chunk)
        return chunk

    def finish(self) -> str | None:
        if self._finished:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "GLM decode stream was finished more than once",
                subsystem="glm4_tokenizer",
            )
        self._finished = True
        complete = self._runtime.decode(
            self._token_ids,
            skip_special_tokens=self._skip_special_tokens,
        )
        streamed = "".join(self._chunks)
        if not complete.startswith(streamed):
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "streaming GLM decode diverged from complete decode",
                subsystem="glm4_tokenizer",
            )
        suffix = complete[len(streamed) :]
        return suffix or None


def _load_tokenizer_engine(path: Path) -> _TokenizerEngine:
    try:
        tokenizers = importlib.import_module("tokenizers")
    except ImportError as exc:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM tokenizer runtime requires the 'tokenizer' optional dependency",
            subsystem="glm4_tokenizer",
        ) from exc
    if getattr(tokenizers, "__version__", None) != _TOKENIZERS_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"GLM tokenizer runtime requires tokenizers {_TOKENIZERS_VERSION}",
            subsystem="glm4_tokenizer",
        )
    try:
        return tokenizers.Tokenizer.from_file(str(path))
    except Exception as exc:
        raise AmsError(
            ErrorCode.BACKEND_FAILURE,
            "tokenizers could not load the admitted GLM-4 tokenizer",
            subsystem="glm4_tokenizer",
        ) from exc


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _compile_template(template: str) -> Any:
    try:
        jinja2 = importlib.import_module("jinja2")
        jinja2_ext = importlib.import_module("jinja2.ext")
        sandbox = importlib.import_module("jinja2.sandbox")
    except ImportError as exc:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM chat rendering requires the 'tokenizer' optional dependency",
            subsystem="glm4_tokenizer",
        ) from exc
    if getattr(jinja2, "__version__", None) != _JINJA2_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"GLM chat rendering requires Jinja2 {_JINJA2_VERSION}",
            subsystem="glm4_tokenizer",
        )
    try:
        environment = sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2_ext.loopcontrols],
        )
        environment.filters["tojson"] = _tojson
        return environment.from_string(template)
    except Exception as exc:
        raise AmsError(
            ErrorCode.BACKEND_FAILURE,
            "the admitted GLM-4 chat template could not be compiled",
            subsystem="glm4_tokenizer",
        ) from exc


def _bounded_chat_inputs(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not messages or len(messages) > _MAX_CHAT_MESSAGES or len(tools) > _MAX_CHAT_TOOLS:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM chat message or tool count is invalid",
            subsystem="glm4_tokenizer",
        )
    copied_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM chat messages must be mappings",
                subsystem="glm4_tokenizer",
            )
        copied = dict(message)
        if copied.get("role") not in {"system", "user", "assistant", "tool"}:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM chat message role is unsupported",
                subsystem="glm4_tokenizer",
            )
        copied_messages.append(copied)
    if any(not isinstance(tool, Mapping) for tool in tools):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM tools must be mappings",
            subsystem="glm4_tokenizer",
        )
    copied_tools = [dict(tool) for tool in tools]
    try:
        source_size = len(
            json.dumps(
                {"messages": copied_messages, "tools": copied_tools},
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM chat inputs must be finite JSON values",
            subsystem="glm4_tokenizer",
        ) from exc
    if source_size > _MAX_CHAT_SOURCE_BYTES:
        raise AmsError(
            ErrorCode.PREFLIGHT_NO_WORKING_SET,
            "GLM chat source exceeds the bounded renderer input",
            subsystem="glm4_tokenizer",
            evidence={"maximum_bytes": _MAX_CHAT_SOURCE_BYTES, "requested_bytes": source_size},
        )
    return copied_messages, copied_tools


class Glm4TokenizerRuntime:
    """Bounded encode/decode and exact pinned chat-template execution."""

    def __init__(
        self,
        assets: Glm4TokenizerAssets,
        engine: _TokenizerEngine,
        compiled_template: Any,
    ) -> None:
        self.assets = assets
        self._engine = engine
        self._compiled_template = compiled_template
        if engine.get_vocab_size(with_added_tokens=True) != assets.tokenizer_vocab_size:
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers loaded an unexpected GLM-4 vocabulary size",
                subsystem="glm4_tokenizer",
            )

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        policy: Glm4TokenizerPolicy = OFFICIAL_GLM4_TOKENIZER_POLICY,
    ) -> Glm4TokenizerRuntime:
        assets = admit_glm4_tokenizer_assets(root, policy=policy)
        return cls(
            assets,
            _load_tokenizer_engine(assets.tokenizer_path),
            _compile_template(assets.template),
        )

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        clear_thinking: bool = True,
    ) -> str:
        copied_messages, copied_tools = _bounded_chat_inputs(messages, tools)
        try:
            rendered = self._compiled_template.render(
                messages=copied_messages,
                tools=copied_tools or None,
                add_generation_prompt=bool(add_generation_prompt),
                enable_thinking=bool(enable_thinking),
                clear_thinking=bool(clear_thinking),
            )
        except Exception as exc:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM chat template rejected the request",
                subsystem="glm4_tokenizer",
            ) from exc
        rendered_size = len(rendered.encode("utf-8"))
        if rendered_size > _MAX_CHAT_SOURCE_BYTES:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "rendered GLM chat exceeds the bounded tokenizer input",
                subsystem="glm4_tokenizer",
                evidence={
                    "maximum_bytes": _MAX_CHAT_SOURCE_BYTES,
                    "requested_bytes": rendered_size,
                },
            )
        return rendered

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM tokenizer input must be text",
                subsystem="glm4_tokenizer",
            )
        source_bytes = len(text.encode("utf-8"))
        if source_bytes > _MAX_CHAT_SOURCE_BYTES:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "GLM tokenizer input exceeds the bounded source size",
                subsystem="glm4_tokenizer",
                evidence={
                    "maximum_bytes": _MAX_CHAT_SOURCE_BYTES,
                    "requested_bytes": source_bytes,
                },
            )
        limit = self.assets.policy.model_max_length if max_tokens is None else max_tokens
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
            or limit > self.assets.policy.model_max_length
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM token limit is invalid",
                subsystem="glm4_tokenizer",
            )
        try:
            token_ids = tuple(self._engine.encode(text, add_special_tokens=False).ids)
        except Exception as exc:
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers failed to encode GLM input",
                subsystem="glm4_tokenizer",
            ) from exc
        if len(token_ids) > limit:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "GLM tokenized input exceeds the admitted context",
                subsystem="glm4_tokenizer",
                evidence={"maximum_tokens": limit, "requested_tokens": len(token_ids)},
            )
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= self.assets.tokenizer_vocab_size
            for token_id in token_ids
        ):
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers emitted an unmapped GLM token",
                subsystem="glm4_tokenizer",
            )
        return token_ids

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        clear_thinking: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        return self.encode(
            self.render_chat(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
            ),
            max_tokens=max_tokens,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        copied = list(token_ids)
        for token_id in copied:
            self._validate_decode_token(token_id)
        try:
            return self._engine.decode(copied, skip_special_tokens=bool(skip_special_tokens))
        except Exception as exc:
            raise AmsError(
                ErrorCode.BACKEND_FAILURE,
                "tokenizers failed to decode GLM output",
                subsystem="glm4_tokenizer",
            ) from exc

    def _validate_decode_token(self, token_id: int) -> None:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= self.assets.tokenizer_vocab_size
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "cannot decode an unmapped GLM model token",
                subsystem="glm4_tokenizer",
            )

    def start_decode_stream(
        self,
        *,
        skip_special_tokens: bool = False,
    ) -> Glm4TokenizerDecodeStream:
        try:
            tokenizers = importlib.import_module("tokenizers")
            decoder = tokenizers.decoders.DecodeStream(
                skip_special_tokens=bool(skip_special_tokens)
            )
        except (ImportError, AttributeError, TypeError) as exc:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "the pinned tokenizers runtime does not expose DecodeStream",
                subsystem="glm4_tokenizer",
            ) from exc
        if getattr(tokenizers, "__version__", None) != _TOKENIZERS_VERSION:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"GLM tokenizer runtime requires tokenizers {_TOKENIZERS_VERSION}",
                subsystem="glm4_tokenizer",
            )
        return Glm4TokenizerDecodeStream(
            self,
            decoder,
            skip_special_tokens=bool(skip_special_tokens),
        )
