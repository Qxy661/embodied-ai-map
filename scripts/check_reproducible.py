#!/usr/bin/env python
"""Prove that the deterministic blocks are actually deterministic.

    python scripts/check_reproducible.py

Each non-volatile block is run twice and its ``results.json`` compared, ignoring
``generated_at_unix`` (the wall clock, which is allowed to move).  Anything else
differing is a failure.

This is worth a separate script rather than a line in CI because "seeded" is a claim
that is easy to believe and easy to be wrong about.  A stray ``np.random`` call, an
unseeded shuffle, or a set iterated in hash order will produce a slightly different
number that nobody notices until a reader cannot reproduce the README.  The check is
cheap; the failure mode it prevents is the one this whole repository exists to avoid.

Volatile blocks are skipped by construction -- they measure the live machine, so
reproducing them to the digit is not a property they have.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "map"

IGNORED_KEYS = {"generated_at_unix"}


def strip_clock(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in IGNORED_KEYS}


def is_volatile(block: pathlib.Path) -> bool:
    meta = block / "meta.yaml"
    if not meta.exists():
        return False
    try:
        return bool((yaml.safe_load(meta.read_text(encoding="utf-8")) or {}).get("volatile"))
    except yaml.YAMLError:
        return False


def run_once(run: pathlib.Path) -> None:
    subprocess.run(
        [sys.executable, str(run)],
        cwd=run.parent,
        capture_output=True,
        check=True,
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    problems: list[str] = []
    checked = skipped = 0

    for run in sorted(MAP.rglob("run.py")):
        block = run.parent
        rel = block.relative_to(MAP).as_posix()
        if is_volatile(block):
            skipped += 1
            print(f"  [skip] {rel:<46} volatile")
            continue

        results = block / "results.json"
        if not results.exists():
            problems.append(f"{rel}: no results.json")
            continue

        run_once(run)
        first = strip_clock(json.loads(results.read_text(encoding="utf-8")))
        run_once(run)
        second = strip_clock(json.loads(results.read_text(encoding="utf-8")))
        checked += 1

        if first == second:
            print(f"  [ok  ] {rel:<46} identical across two runs")
            continue

        # Report the specific keys that moved, so the fix is obvious.
        diffs = []
        for key in sorted(set(first) | set(second)):
            if first.get(key) != second.get(key):
                if key == "metrics" and isinstance(first.get(key), dict):
                    for m in sorted(set(first[key]) | set(second[key])):
                        if first[key].get(m) != second[key].get(m):
                            diffs.append(
                                f"metrics.{m}: {first[key].get(m)} != {second[key].get(m)}"
                            )
                else:
                    diffs.append(key)
        problems.append(f"{rel}: results.json changed between runs -> {'; '.join(diffs[:6])}")
        print(f"  [FAIL] {rel:<46} NOT reproducible")

    print(f"\ncheck_reproducible: {checked} block(s) reproduced, {skipped} skipped (volatile)")
    if problems:
        print(f"\ncheck_reproducible: FAILED ({len(problems)} problem(s))", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nA deterministic block must produce identical results.json twice.\n"
            "Usual causes: an unseeded RNG, iterating a set, or reading the clock.\n"
            "If the block genuinely measures the live machine, declare volatile: true.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
