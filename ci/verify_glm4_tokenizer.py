"""Qualify the pinned GLM-4 tokenizer against deterministic prompt vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import tokenizers

from ams.integrations.glm4_tokenizer import (
    OFFICIAL_GLM4_TOKENIZER_POLICY,
    Glm4TokenizerRuntime,
)

_CASES: tuple[
    tuple[
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        bool,
        int,
        str,
    ],
    ...,
] = (
    (
        "hello",
        [{"role": "user", "content": "Hello, world!"}],
        [],
        True,
        9,
        "ced86f38ce5c308f21f5649bf5d003b4e5434ac99e7ae7d62f2730da57712956",
    ),
    (
        "tool",
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        False,
        164,
        "064e788c4131f6ba3592a8695ad1657fc0feaf9b49c781b32b8e13592e68f3a6",
    ),
    (
        "history",
        [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "reasoning_content": "hidden",
                "content": "answer",
            },
            {"role": "user", "content": [{"type": "text", "text": "Next"}]},
        ],
        [],
        True,
        11,
        "6b62d189514fe54e8877d1ba0cb76a88c4afe1a68a05b91ab252272c808eefcc",
    ),
)
_HELLO_IDS = (154822, 154824, 154827, 9703, 11, 1879, 0, 154828, 154841)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Pinned GLM-4.7-Flash asset directory")
    parser.add_argument(
        "--transformers-oracle",
        action="store_true",
        help="also require exact parity with installed Transformers 5.12.0",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    runtime = Glm4TokenizerRuntime.from_root(arguments.root)
    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError("qualification requires tokenizers 0.22.2")

    oracle = None
    oracle_version = None
    if arguments.transformers_oracle:
        import transformers
        from transformers import PreTrainedTokenizerFast

        if transformers.__version__ != "5.12.0":
            raise RuntimeError("qualification requires Transformers 5.12.0")
        oracle = PreTrainedTokenizerFast.from_pretrained(
            arguments.root,
            local_files_only=True,
        )
        oracle_version = transformers.__version__

    evidence: list[dict[str, Any]] = []
    for name, messages, tools, enable_thinking, expected_count, expected_hash in _CASES:
        rendered = runtime.render_chat(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        token_ids = runtime.encode(rendered)
        actual_hash = hashlib.sha256(rendered.encode()).hexdigest()
        if len(token_ids) != expected_count or actual_hash != expected_hash:
            raise RuntimeError(f"{name} prompt vector changed")
        if runtime.decode(token_ids, skip_special_tokens=False) != rendered:
            raise RuntimeError(f"{name} prompt does not round-trip")
        if name == "hello" and token_ids != _HELLO_IDS:
            raise RuntimeError("hello token IDs changed")

        oracle_equal = None
        if oracle is not None:
            oracle_text = oracle.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                clear_thinking=True,
            )
            oracle_ids = tuple(
                oracle.apply_chat_template(
                    messages,
                    tools=tools or None,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    clear_thinking=True,
                )
            )
            oracle_equal = oracle_text == rendered and oracle_ids == token_ids
            if not oracle_equal:
                raise RuntimeError(f"{name} differs from the Transformers oracle")

        evidence.append(
            {
                "name": name,
                "oracle_equal": oracle_equal,
                "rendered_sha256": actual_hash,
                "token_count": len(token_ids),
            }
        )

    print(
        json.dumps(
            {
                "cases": evidence,
                "repository": OFFICIAL_GLM4_TOKENIZER_POLICY.repository,
                "revision": OFFICIAL_GLM4_TOKENIZER_POLICY.revision,
                "tokenizer_version": tokenizers.__version__,
                "transformers_version": oracle_version,
                "unmapped_model_token_ids": list(runtime.assets.unmapped_model_token_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
