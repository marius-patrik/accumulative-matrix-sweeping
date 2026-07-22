import json
from pathlib import Path

from jsonschema.validators import Draft202012Validator


def test_all_repository_schemas_are_valid_draft_2020_12() -> None:
    schema_root = Path(__file__).parents[2] / "schemas"
    schemas = sorted(schema_root.glob("*.schema.json"))
    assert schemas
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
