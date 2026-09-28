#!/usr/bin/env python
"""Run every block's ``run.py`` and produce ``results/summary.json``.

    python scripts/run_all.py                 # everything
    python scripts/run_all.py 03 06           # only rings whose name starts with 03 / 06
    python scripts/run_all.py --skip-volatile # leave the live-measurement blocks alone

Each block runs as its own subprocess so that one crash neither hides nor aborts the
others; the exit code is non-zero if any block failed.  Blocks are deliberately
cheap -- the whole map is meant to finish in well under a minute on a laptop CPU,
because a survey nobody can afford to re-run is a survey nobody will trust.

``--skip-volatile`` exists for CI.  A block that measures the live machine records
numbers that are true of *this* host, so re-running it on a different machine
rewrites its ``results.json`` with values its committed README no longer matches.
CI therefore regenerates the deterministic blocks only, and leaves the volatile
ones checked against their committed results.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "map"
RESULTS = ROOT / "results"


def is_volatile(block: pathlib.Path) -> bool:
    meta = block / "meta.yaml"
    if not meta.exists():
        return False
    try:
        return bool((yaml.safe_load(meta.read_text(encoding="utf-8")) or {}).get("volatile"))
    except yaml.YAMLError:
        return False


def discover(filters: list[str], skip_volatile: bool) -> list[pathlib.Path]:
    runs = sorted(MAP.rglob("run.py"))
    if skip_volatile:
        runs = [r for r in runs if not is_volatile(r.parent)]
    if filters:
        runs = [r for r in runs if any(f in r.parts for f in filters)]
    return runs


def main(argv: list[str]) -> int:
    skip_volatile = "--skip-volatile" in argv
    filters = [a for a in argv if not a.startswith("-")]
    runs = discover(filters, skip_volatile)
    if not runs:
        print("run_all: nothing matched", file=sys.stderr)
        return 1
    if skip_volatile:
        n_skipped = len(list(MAP.rglob("run.py"))) - len(runs)
        print(f"run_all: skipping {n_skipped} volatile block(s)")

    entries = []
    failures = []
    t_start = time.perf_counter()

    for run in runs:
        block = run.parent
        rel = block.relative_to(MAP).as_posix()
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(run)],
            cwd=block,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        entries.append({"block": rel, "ok": ok, "seconds": round(dt, 3)})
        tail = (proc.stdout or "").strip().splitlines()
        summary = tail[-1] if tail else ""
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {rel:<46} {dt * 1000:7.0f} ms  {summary[:70]}")
        if not ok:
            failures.append((rel, proc.stderr))
        if proc.stderr and ok:
            print(f"         stderr: {proc.stderr.strip().splitlines()[-1][:100]}")

    total = time.perf_counter() - t_start
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "blocks": entries,
                "total_seconds": round(total, 3),
                "failed": [f[0] for f in failures],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nrun_all: {len(entries) - len(failures)}/{len(entries)} ok in {total:.2f} s")
    for rel, err in failures:
        print(f"\n--- {rel} ---\n{err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
