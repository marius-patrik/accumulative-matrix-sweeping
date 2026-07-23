from __future__ import annotations

import pytest

from ams.errors import AmsError, ErrorCode
from ci.audit_glm52_source import _normalize_lfs_siblings, _strict_json


def test_glm52_audit_accepts_only_strict_json() -> None:
    assert _strict_json(b'{"sha":"revision"}', label="fixture") == {"sha": "revision"}
    with pytest.raises(AmsError, match="strict JSON") as duplicate:
        _strict_json(b'{"sha":"first","sha":"second"}', label="fixture")
    assert duplicate.value.code is ErrorCode.INVALID_PACKAGE
    with pytest.raises(AmsError, match="strict JSON") as nonfinite:
        _strict_json(b'{"size":NaN}', label="fixture")
    assert nonfinite.value.code is ErrorCode.INVALID_PACKAGE


def test_glm52_audit_normalizes_exact_lfs_identity() -> None:
    digest = "a" * 64
    assert _normalize_lfs_siblings(
        {
            "siblings": [
                {
                    "rfilename": "model-00001-of-00001.safetensors",
                    "lfs": {"size": 123, "sha256": digest},
                },
                {"rfilename": "config.json"},
            ]
        }
    ) == {"model-00001-of-00001.safetensors": (123, digest)}


@pytest.mark.parametrize(
    "lfs",
    [
        {"size": True, "sha256": "a" * 64},
        {"size": 0, "sha256": "a" * 64},
        {"size": 123, "sha256": "not-a-digest"},
    ],
)
def test_glm52_audit_rejects_malformed_lfs_identity(lfs: dict[str, object]) -> None:
    with pytest.raises(AmsError, match="LFS metadata") as caught:
        _normalize_lfs_siblings(
            {
                "siblings": [
                    {
                        "rfilename": "model-00001-of-00001.safetensors",
                        "lfs": lfs,
                    }
                ]
            }
        )
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
