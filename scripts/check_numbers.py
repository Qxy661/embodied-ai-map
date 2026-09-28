#!/usr/bin/env python
"""Fail the build if a README cites a number that no run produced.

This is the mechanism that makes the repository a *survey* rather than a blog post
with a code folder bolted on.  Prose and numbers are not allowed to drift apart:
every quantity printed in a block's ``README.md`` must be a correct rounding of
something inside the sibling ``results.json``.

Three ways a number is allowed to appear:

1. It matches a value in ``results.json`` (rounding at the precision it is written).
2. It sits on a line carrying the citation marker ``!CITE``, meaning it came from a
   paper or vendor blog and is quoted, not measured.
3. It is listed under ``literals:`` in the block's ``meta.yaml`` -- for genuinely
   structural numbers (a section index, ``H=50`` named in prose) that are not
   results at all.

Anything else is an error.  The fix is never to edit the checker.

The one concession to physics: a block that *measures the live machine* (timing,
jitter) cannot reproduce to the digit, because the machine is not the same machine
twice.  Those blocks declare ``volatile: true`` in ``meta.yaml`` and are checked
approximately instead of exactly.  Everything simulated or seeded stays exact.

What that concession does and does not buy, stated plainly, because a weakened
check that oversells itself is worse than no check:

  * it does catch a number that came from nowhere.  ``7777``, ``12345`` and
    ``55.5`` all fail against the jitter block.
  * it does *not* catch a fabrication that happens to land near a real value.
    ``400`` passes, because the block really does measure a p50 near 308 us.
    No tolerance-based check can separate those two cases, and pretending
    otherwise would be the dishonest move.
  * a volatile block should therefore declare ``citable:`` -- the few metric keys
    its prose is allowed to quote.  Strength here is set by how dense the
    candidate set is: an approximate window around all ~28 metrics of a block
    covers essentially the whole number line and catches nothing, which is exactly
    how ``55.5`` slipped through an earlier version of this file.
  * the durable guarantee for a volatile block is not this script.  It is that the
    numbers are *emitted by run.py* and that the diff is reviewable.  Prefer
    reporting quantities that are stable across runs -- pooled samples, or
    deliberately conservative floors -- over raw per-run output.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "map"

CITE_MARKER = "!CITE"
# Numbers as they appear in prose: optional sign, digits, optional decimal part.
# Bare integers that are part of an identifier (e.g. "T1", "03-decision") are not
# matched because the boundary classes require a non-word char (or string edge).
NUM_RE = re.compile(r"(?<![\w.\-])(-?\d+(?:\.\d+)?)(?![\w.\-])")


def load_numbers(obj, acc: set[float]) -> set[float]:
    """Recursively collect every number in a JSON payload."""
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            load_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            load_numbers(v, acc)
    elif isinstance(obj, str):
        # numeric strings inside payloads still count as sources
        for m in NUM_RE.finditer(obj):
            acc.add(float(m.group(1)))
    return acc


def decimals_of(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def rel(path: pathlib.Path) -> str:
    """Repo-relative for readability, absolute for anything outside the repo.

    ``Path.relative_to`` raises rather than falling back, which turned a tidy error
    message into a crash for blocks outside ``ROOT`` (e.g. the checker's own tests,
    which build throwaway blocks in a temp directory).
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def matches(x: float, candidates: set[float], decimals: int) -> bool:
    """True if some candidate y rounds to x at the precision x was written."""
    for y in candidates:
        if abs(round(y, decimals) - x) < 1e-9:
            return True
        # percentages: README says 42.0, results hold 0.42
        if abs(round(y * 100.0, decimals) - x) < 1e-9:
            return True
        # ...and the reverse, for rates stored as percentages
        if abs(round(y / 100.0, decimals) - x) < 1e-9:
            return True
    return False


# A block whose headline result is a *live measurement* (timing, jitter) cannot be
# reproduced to the digit -- the machine is not the same machine twice.  For those,
# declared with ``volatile: true`` in meta.yaml, we check approximate agreement
# instead of exact rounding.  See docs/00.
#
# Two things keep that concession from swallowing the invariant:
#
#   * the tolerance is small (2x, not "order of magnitude").  A wide tolerance
#     against a dense candidate set accepts almost any number: an early version at
#     10x let a fabricated "7777" through, because some real result always sits
#     within an order of magnitude.
#   * candidates are restricted to ``metrics``.  The per-pass ``tables`` hold tens
#     of raw samples spanning four orders of magnitude, which makes the net dense
#     enough to catch nothing.  A volatile block therefore has to promote anything
#     it wants to quote into ``metrics`` -- which is the right pressure, because it
#     forces the block to name its headline numbers instead of citing raw dumps.
#
# The matching consequence is on the block: it should report statistics that are
# actually stable across runs.  Pooling samples across passes is the usual way.
VOLATILE_TOLERANCE = 1.5

# The percentage / fraction rescaling stays *exact* even for volatile blocks.  The
# measurement is noisy; the unit conversion is not -- a block that writes "83.14%"
# for a stored 0.8314 is converting, not approximating.  Letting the rescale inherit
# the loose tolerance is what broke an earlier version: jitter_budget_us = 100
# rescaled to 10000, and a 2x window around it swallowed a fabricated "7777".
SCALE_TOLERANCE = 1.02


def matches_loosely(x: float, candidates: set[float]) -> bool:
    """Approximate agreement on the value, exact agreement on the unit."""
    if x == 0.0:
        return any(abs(y) < 1e-9 for y in candidates)
    for y in candidates:
        if y != 0 and (1.0 / VOLATILE_TOLERANCE) <= abs(x / y) <= VOLATILE_TOLERANCE:
            return True
        # percent <-> fraction, held to the exact-conversion tolerance
        for scale in (100.0, 0.01):
            sy = y * scale
            if sy != 0 and (1.0 / SCALE_TOLERANCE) <= abs(x / sy) <= SCALE_TOLERANCE:
                return True
    return False


def check_block(block: pathlib.Path) -> tuple[list[str], int]:
    readme = block / "README.md"
    results = block / "results.json"
    if not readme.exists():
        return [], 0
    if not results.exists():
        return [f"{rel(block)}: has README.md but no results.json (run the block)"], 0

    payload = json.loads(results.read_text(encoding="utf-8"))

    meta_path = block / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta = meta or {}
    literals: set[float] = set()
    for v in meta.get("literals", []) or []:
        literals.add(float(v))
    volatile = bool(meta.get("volatile", False))

    # Volatile blocks are checked against metrics only -- see VOLATILE_TOLERANCE.
    # If the block also declares ``citable:``, the candidate set narrows to those
    # keys.  That is not bookkeeping: the strength of an approximate check is set
    # by how dense the candidate set is, and a block's full metrics dict is dense
    # enough to cover the whole number line.  See the note on VOLATILE_TOLERANCE.
    metrics = payload.get("metrics", {})
    if volatile:
        declared = meta.get("citable")
        if declared:
            unknown = [k for k in declared if k not in metrics]
            if unknown:
                return [
                    f"{rel(block)}/meta.yaml: citable names unknown metric(s): {', '.join(unknown)}"
                ], 0
            metrics = {k: metrics[k] for k in declared}
        candidates = load_numbers(metrics, set())
    else:
        candidates = load_numbers(payload, set())

    errors: list[str] = []
    loose_hits = 0
    for lineno, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        if CITE_MARKER in line:
            continue
        for m in NUM_RE.finditer(line):
            token = m.group(1)
            x = float(token)
            d = decimals_of(token)
            if not volatile:
                if matches(x, candidates, d) or matches(x, literals, d):
                    continue
            else:
                # literals stay exact even here: they are structural constants
                # (an index, a named rate), not measurements.  Letting them into
                # the loose net is how a fabricated "7777" once slipped through --
                # it sat within the window around the literal 10000.
                if matches_loosely(x, candidates) or matches(x, literals, d):
                    loose_hits += 1
                    continue
            errors.append(
                f"{rel(readme)}:{lineno}: {token!r} is not in results.json "
                f"and is not marked {CITE_MARKER}\n      > {line.strip()[:110]}"
            )
    return errors, loose_hits


def check_orphans() -> list[str]:
    """Every results.json must be regenerated by a run.py, and be committed fresh."""
    errors = []
    for results in sorted(MAP.rglob("results.json")):
        if not (results.parent / "run.py").exists():
            errors.append(f"{rel(results)}: no run.py next to it")
    return errors


def discover_blocks() -> list[pathlib.Path]:
    """A *block* is a directory containing ``run.py``.

    That definition matters: a ring directory (``map/03-decision/``) carries its own
    ``README.md`` holding the content that can only be cited, never measured.  It has
    no results of its own and is deliberately *not* a block.  Keying on ``run.py``
    rather than on ``README.md`` keeps the two kinds of file distinct instead of
    requiring the ring overviews to be named something other than README.
    """
    return sorted(p.parent for p in MAP.rglob("run.py"))


def main() -> int:
    # Error lines quote the offending prose, which is Chinese in this repo; without
    # this the console mangles them on a non-UTF-8 code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    blocks = discover_blocks()
    if not blocks:
        print("check_numbers: no blocks found", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    loose_blocks: list[str] = []
    for b in blocks:
        errs, loose = check_block(b)
        all_errors.extend(errs)
        if loose:
            loose_blocks.append(f"{rel(b)} ({loose})")
    all_errors.extend(check_orphans())

    if all_errors:
        print(f"check_numbers: FAILED ({len(all_errors)} problem(s))\n", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nNumbers in prose must come from a run.  Re-run the block, or mark the\n"
            f"line {CITE_MARKER} if it is quoted from a paper, or add it to literals: in meta.yaml.",
            file=sys.stderr,
        )
        return 1

    print(f"check_numbers: OK -- {len(blocks)} block(s), every cited number traced to a run")
    if loose_blocks:
        print(
            f"  volatile blocks checked to order of magnitude (x{VOLATILE_TOLERANCE:g}), "
            f"not exact rounding: {', '.join(loose_blocks)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
