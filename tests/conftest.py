"""Put the project's package directories on sys.path.

The modules import each other by bare name (`from tokenizer import ...`), which
works when a script is run from inside its own directory but not when pytest is
run from the repo root. This adds both directories so either works.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for package in ("regression", "tiny_llm"):
    path = str(ROOT / package)
    if path not in sys.path:
        sys.path.insert(0, path)
