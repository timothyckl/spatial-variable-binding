"""Repository and TransformerLens import-path bootstrap.

Legacy models use the vendored ``HookedVLTransformer``. Qwen3-VL uses the
installed TransformerLens ``TransformerBridge``, whose native architecture
adapter preserves DeepStack and interleaved MRoPE. Experiments run one model
per process, so the requested ``--model`` selects the backend before the first
TransformerLens import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TLENS_ROOT = REPO_ROOT / "third_party" / "TransformerLens"
_QWEN3_NAMES = {
    "qwen3vl",
    "qwen3-vl-8b",
    "qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-8b-instruct",
}


def _requested_backend() -> str:
    override = os.environ.get("SPATIAL_TLENS_BACKEND")
    if override:
        if override not in {"legacy", "bridge"}:
            raise ValueError("SPATIAL_TLENS_BACKEND must be 'legacy' or 'bridge'")
        return override

    for index, arg in enumerate(sys.argv):
        if arg == "--model" and index + 1 < len(sys.argv):
            return "bridge" if sys.argv[index + 1].lower() in _QWEN3_NAMES else "legacy"
        if arg.startswith("--model="):
            return "bridge" if arg.split("=", 1)[1].lower() in _QWEN3_NAMES else "legacy"
    return "legacy"


def setup() -> None:
    """Put the repository and the selected TransformerLens backend on sys.path."""
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    vendored = str(TLENS_ROOT)
    if _requested_backend() == "bridge":
        while vendored in sys.path:
            sys.path.remove(vendored)
    elif vendored not in sys.path:
        sys.path.insert(0, vendored)
