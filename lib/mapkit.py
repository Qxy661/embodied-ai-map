"""Shared helpers for every block under ``map/``.

The one rule this module exists to enforce: **a block never writes its results by
hand.**  ``run.py`` computes, :func:`emit` serialises.  ``scripts/check_numbers.py``
then reads the ``results.json`` back and verifies that every number printed in the
sibling ``README.md`` is a correct rounding of something in there.  If you edit a
number in a README without re-running the block, CI fails.

Everything here is numpy + stdlib on purpose -- no torch, no scipy.  The blocks are
minimal reproductions of a *mechanism*, not of a production stack, and a dependency
that can be avoided is a dependency that cannot break the reader's environment.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "map"


def rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator.  Blocks must never touch the global random state."""
    return np.random.default_rng(seed)


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def emit(
    block_dir: str | pathlib.Path,
    metrics: dict,
    tables: dict | None = None,
    *,
    seed: int = 0,
    notes: str | None = None,
):
    """Serialise a block's results next to its ``README.md``.

    ``metrics`` holds the scalars the README cites; ``tables`` holds the swept
    rows.  Both are flattened by the checker, so nesting is free.
    """
    block_dir = pathlib.Path(block_dir)
    payload = {
        "block": str(block_dir.relative_to(ROOT).as_posix()),
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_rev": _git_rev(),
        "generated_at_unix": int(time.time()),
        "metrics": metrics,
        "tables": tables or {},
    }
    if notes:
        payload["notes"] = notes
    out = block_dir / "results.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def digest(obj) -> str:
    """Stable short hash of a result payload, for the reproducibility ledger."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def timer(fn, *, repeats: int = 1):
    """Run ``fn`` ``repeats`` times, return (last_result, min_seconds).

    min, not mean: on a laptop the only thing measurement noise can do is make a
    run slower.  See docs/00 for why this matters more than it looks.
    """
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion.

    Closed form, so no scipy.  ``z`` defaults to the two-sided 95% normal quantile.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def humanise(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def main_guard():
    """Blocks print a one-line summary so ``run_all.py`` output stays readable."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
