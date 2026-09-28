#!/usr/bin/env python
"""Aggregate every block's results.json into results/ledger.{json,md}.

The ledger is the audit trail: for each block it records what was measured, the
seed, the numpy version and the git revision that produced it.  If a number in the
README cannot be traced here, scripts/check_numbers.py will say so.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "map"
RESULTS = ROOT / "results"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for rj in sorted(MAP.rglob("results.json")):
        payload = json.loads(rj.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "block": payload["block"],
                "metrics": metrics,
                "n_metrics": len(metrics),
                "n_tables": len(payload.get("tables", {})),
                "seed": payload.get("seed"),
                "numpy": payload.get("numpy"),
                "git_rev": payload.get("git_rev"),
                "generated_at_unix": payload.get("generated_at_unix"),
            }
        )

    (RESULTS / "ledger.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# 结果台账",
        "",
        "由 `python scripts/collect.py` 生成，请勿手改。",
        "每一行是一个可运行块，以及它产出的关键量。README 里出现的每个数字都必须能在",
        "这里的 `metrics`（或块自己的 `results.json` 的 `tables`）里找到，否则 CI 不过。",
        "",
        "| 块 | 关键量 | 值 |",
        "|---|---|---|",
    ]
    for r in rows:
        if not r["metrics"]:
            continue
        first = True
        for k, v in sorted(r["metrics"].items()):
            if isinstance(v, float):
                v = f"{v:.6g}"
            cell_block = f"`{r['block']}`" if first else ""
            lines.append(f"| {cell_block} | `{k}` | {v} |")
            first = False

    (RESULTS / "ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"collect: wrote ledger for {len(rows)} block(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
