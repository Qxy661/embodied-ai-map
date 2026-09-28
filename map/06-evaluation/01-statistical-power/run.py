#!/usr/bin/env python
"""How many robot trials does it take to tell two policies apart?

Success rate is a binomial proportion.  Robot evaluations are routinely reported
with 20-30 trials and no confidence interval, which -- as this block quantifies --
cannot distinguish a good policy from a much better one.  The point is not that
anyone is lying; it is that the standard experimental design in this field has a
resolution far coarser than the effects people argue about.

Everything here is closed-form or simulated from the binomial, so there is no model
of robot behaviour to disagree with: the arithmetic is the whole argument.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

P_A = 0.70  # the weaker policy
P_B = 0.92  # the stronger one -- a 22 point gap, which is enormous
NS = [10, 20, 30, 50, 100, 200, 500]
M = 40000  # simulated experiments per cell
ALPHA_Z = 1.959963984540054  # two-sided 95%
POWER_TARGET = 0.80  # the conventional target we solve for


def power(n: int, p_a: float, p_b: float, m: int, rng: np.random.Generator) -> float:
    """Simulated power of a pooled two-proportion z-test at n trials per arm."""
    x_a = rng.binomial(n, p_a, m)
    x_b = rng.binomial(n, p_b, m)
    p_pool = (x_a + x_b) / (2 * n)
    se = np.sqrt(p_pool * (1 - p_pool) * 2.0 / n)
    se = np.where(se > 0, se, np.inf)
    z = (x_b - x_a) / n / se
    return float((np.abs(z) > ALPHA_Z).mean())


def main() -> int:
    rng = mapkit.rng(0)

    rows = []
    for n in NS:
        _, lo_a, hi_a = mapkit.wilson_interval(round(P_A * n), n, ALPHA_Z)
        _, lo_b, hi_b = mapkit.wilson_interval(round(P_B * n), n, ALPHA_Z)
        rows.append(
            {
                "N": n,
                "ci_halfwidth_pp_p70": round((hi_a - lo_a) / 2 * 100, 2),
                "ci_halfwidth_pp_p92": round((hi_b - lo_b) / 2 * 100, 2),
                "power": round(power(n, P_A, P_B, M, rng), 4),
            }
        )

    by_n = {r["N"]: r for r in rows}

    # How many trials per arm to reach 80% power on this 22 point gap?
    n_needed = None
    for n in range(5, 400):
        if power(n, P_A, P_B, 4000, rng) >= POWER_TARGET:
            n_needed = n
            break

    at20 = by_n[20]
    _, lo20, hi20 = mapkit.wilson_interval(round(P_A * 20), 20, ALPHA_Z)
    metrics = {
        "p_a": P_A,
        "p_b": P_B,
        "true_gap_pp": round((P_B - P_A) * 100, 1),
        "simulated_experiments_per_cell": M,
        "ci_lo_pct_p70_at_N20": round(lo20 * 100, 1),
        "ci_hi_pct_p70_at_N20": round(hi20 * 100, 1),
        "ci_halfwidth_pp_at_N20": at20["ci_halfwidth_pp_p70"],
        "power_at_N20": at20["power"],
        "power_target": POWER_TARGET,
        "n_needed_for_80pct_power": n_needed,
        "power_at_N100": by_n[100]["power"],
        "power_at_N500": by_n[500]["power"],
        "ci_halfwidth_pp_at_N500": by_n[500]["ci_halfwidth_pp_p70"],
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"sweep": rows},
        notes=(
            "Power is the simulated rejection rate of a pooled two-proportion z-test "
            f"over {M} draws per cell. CI half-width is Wilson."
        ),
    )
    print(
        f"@N=20: CI +-{at20['ci_halfwidth_pp_p70']:.1f}pp, power on a {metrics['true_gap_pp']:.0f}pp "
        f"gap = {at20['power']:.3f} | need N={n_needed} for 0.80 -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
