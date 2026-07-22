#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "error: latexmk is required" >&2
  exit 2
fi

# Some container images leave /usr/bin/bibtex as a broken alternatives link while
# retaining the actual binary as bibtex.original. Prefer the normal executable,
# then add a temporary shim without modifying the system.
TMP_BIN=""
if ! bibtex --version >/dev/null 2>&1; then
  if [[ -x /usr/bin/bibtex.original ]]; then
    TMP_BIN="$(mktemp -d)"
    trap 'rm -rf "$TMP_BIN"' EXIT
    ln -s /usr/bin/bibtex.original "$TMP_BIN/bibtex"
    export PATH="$TMP_BIN:$PATH"
  else
    echo "error: bibtex is required" >&2
    exit 2
  fi
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

python3 - <<'PY'
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

root = Path.cwd()
log = (root / "main.log").read_text(errors="replace")

fatal_patterns = {
    "overfull box": r"Overfull \\hbox|Overfull \\vbox",
    "undefined reference": r"LaTeX Warning: (?:Reference|Citation).*undefined",
    "undefined control sequence": r"Undefined control sequence",
    "duplicate destination": r"destination with the same identifier|multiply defined",
}
failures = [name for name, pattern in fatal_patterns.items() if re.search(pattern, log)]
if failures:
    raise SystemExit("paper validation failed: " + ", ".join(failures))

for path in sorted((root / "schemas").glob("*.json")):
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise SystemExit(f"{path}: expected JSON Schema draft 2020-12")

text = subprocess.check_output(["pdftotext", "-layout", "main.pdf", "-"], text=True)
for marker in ("??", "AUTHOR INFORMATION TO BE SUPPLIED", "AMSseparates", "AMSlinear"):
    if marker in text:
        raise SystemExit(f"rendered PDF contains unresolved marker: {marker!r}")

print("validated: main.pdf and", len(list((root / "schemas").glob("*.json"))), "JSON schemas")
PY

pdfinfo main.pdf | sed -n '1,22p'
