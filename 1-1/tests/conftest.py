from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "Code"
DOCFLOW_SRC = CODE_ROOT / "docflow_src"

for path in (PROJECT_ROOT, CODE_ROOT, DOCFLOW_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
