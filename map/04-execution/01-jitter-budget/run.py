#!/usr/bin/env python
"""What a 1 kHz control loop actually costs in jitter, measured on this machine.

The joint servo loop assumed by everything upstream -- chunking, RTC, the whole
latency budget -- runs at 1000 Hz, i.e. a 1000 us period.  This block measures
whether it holds that schedule on a general-purpose desktop OS with no real-time
patch, and what a busy-wait buys over sleeping.

Two wake-up strategies, each with and without CPU contention:

  sleep   give the core back between ticks (courteous, coarse)
  spin    burn the core to hit the deadline (precise, antisocial)

Two design decisions here were forced by the first version of this block failing,
and both are the point of it:

1.  **Repeats, and pooled statistics.**  A single pass produced a p99 ratio between
    the two strategies that moved across two orders of magnitude between runs --
    so a single pass cannot rank them at all.  The block now runs REPEATS passes
    and pools the per-tick samples, which is what makes its headline numbers
    reproducible.  The *instability itself* is reported (p99_ratio_swing_floor).

2.  **Score the fraction over budget, not the p99.**  The p99 of 1200 ticks is the
    12th-worst sample; one outlier decides it.  The over-budget fraction uses every
    sample.  They disagree, and the disagreement is the finding.

Because this is wall-clock measurement on a general-purpose OS it is declared
``volatile: true`` in meta.yaml, and check_numbers.py checks it approximately
rather than exactly.  The absolutes are host-specific and so is the sleep-vs-spin
ordering -- on an isolated core under PREEMPT_RT the busy-wait is the one that
wins.  What transfers is the method: score the tail, score it under load, and do
not trust a single order statistic.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

PERIOD_US = 1000.0  # 1 kHz
JITTER_BUDGET_US = 100.0  # the conventional "10% of period" allowance
N_TICKS = 1200  # per condition per pass
N_CONTENDERS = 8
REPEATS = 3

# Lower bounds the README is allowed to quote.  These are *claims*, checked at the
# end of the run, not measurements -- see the comment at obf_ratio_floor in main().
# Each carries real margin over every session observed so far, and the assertion
# below is what keeps them honest: if the machine ever drifts past one of these,
# the run fails instead of silently reporting a weaker number than the prose says.
OBF_RATIO_FLOOR = 2  # spin's idle over-budget fraction is >=2x rarer than sleep's
P50_RATIO_FLOOR = 1000  # ... and its median lateness is >=1000x smaller
P99_SWING_FLOOR = 3  # the p99 ranking is not stable to better than 3x between passes

CONDITIONS = ("sleep_idle", "spin_idle", "sleep_contended", "spin_contended")


def _contend(stop: threading.Event) -> None:
    while not stop.is_set():
        pass


def measure(spin: bool) -> np.ndarray:
    """Return per-tick lateness in microseconds (negative = early)."""
    lateness = np.empty(N_TICKS)
    period = PERIOD_US / 1e6
    t_next = time.perf_counter() + period
    for i in range(N_TICKS):
        if spin:
            while time.perf_counter() < t_next:
                pass
        else:
            while True:
                remaining = t_next - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(remaining * 0.5)
        lateness[i] = (time.perf_counter() - t_next) * 1e6
        t_next += period
    return lateness


def summarise(x: np.ndarray) -> dict:
    """The three statistics that disagree with each other."""
    return {
        "p50_us": round(float(np.percentile(x, 50)), 2),
        "p99_us": round(float(np.percentile(x, 99)), 2),
        # Uses every sample, so it is the one that survives repeated runs.
        "over_budget_frac": round(float(np.mean(x > JITTER_BUDGET_US)), 4),
    }


def one_pass() -> dict[str, np.ndarray]:
    """One full sweep: {sleep,spin} x {idle,contended}, raw per-tick lateness."""
    out = {"sleep_idle": measure(False), "spin_idle": measure(True)}

    stop = threading.Event()
    workers = [
        threading.Thread(target=_contend, args=(stop,), daemon=True) for _ in range(N_CONTENDERS)
    ]
    for w in workers:
        w.start()
    try:
        time.sleep(0.05)
        out["sleep_contended"] = measure(False)
        out["spin_contended"] = measure(True)
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=1.0)
    return out


def main() -> int:
    passes = [one_pass() for _ in range(REPEATS)]

    # Pooled across passes: 3 x 1200 samples per condition.  This is what makes the
    # headline numbers reproducible; see the module docstring.
    pooled = {c: np.concatenate([p[c] for p in passes]) for c in CONDITIONS}
    stats = {c: summarise(v) for c, v in pooled.items()}

    def median_across_passes(cond: str, field: str) -> float:
        return round(float(np.median([summarise(p[cond])[field] for p in passes])), 4)

    # Per-pass p99 ratio: the spread of this is the block's cautionary tale.
    ratios = [
        summarise(p["sleep_idle"])["p99_us"] / max(summarise(p["spin_idle"])["p99_us"], 1e-9)
        for p in passes
    ]
    swing = max(ratios) / max(min(ratios), 1e-9)

    metrics = {
        "period_us": PERIOD_US,
        "jitter_budget_us": JITTER_BUDGET_US,
        "n_ticks_per_pass": N_TICKS,
        "n_samples_pooled": N_TICKS * REPEATS,
        "n_contenders": N_CONTENDERS,
        "repeats": REPEATS,
        # --- the statistic that holds up: fraction of ticks over budget -------
        "sleep_idle_over_budget_frac": stats["sleep_idle"]["over_budget_frac"],
        "spin_idle_over_budget_frac": stats["spin_idle"]["over_budget_frac"],
        "obf_ratio": round(
            stats["sleep_idle"]["over_budget_frac"]
            / max(stats["spin_idle"]["over_budget_frac"], 1e-9),
            1,
        ),
        # The citable form of the ratio.  A *floor* has to be a claim that holds
        # every session -- that is the whole point -- so it cannot itself be a
        # measurement.  Emitting min(FLOOR, int(ratio)) looked conservative but
        # was not: when a session came back at 3.5x the block emitted 3 and the
        # prose, which said 5, stopped matching.  These are now constants the run
        # *asserts* against, so the claim is stable by construction and a session
        # that ever violates it fails loudly instead of quietly redefining itself.
        "obf_ratio_floor": OBF_RATIO_FLOOR,
        # --- the statistic that flatters spin and misleads --------------------
        "sleep_idle_p50_us": stats["sleep_idle"]["p50_us"],
        "spin_idle_p50_us": stats["spin_idle"]["p50_us"],
        "p50_ratio": round(
            stats["sleep_idle"]["p50_us"] / max(stats["spin_idle"]["p50_us"], 1e-9), 1
        ),
        # Same as obf_ratio_floor: an asserted constant, not a capped measurement.
        # The raw ratio has run 1369..1865 across sessions, so 1000 has margin.
        "p50_ratio_floor": P50_RATIO_FLOOR,
        # --- the statistic that cannot rank the two ---------------------------
        "p99_ratio_min": round(min(ratios), 2),
        "p99_ratio_max": round(max(ratios), 2),
        # The raw swing ranges ~4x..12x between sessions.  "The ranking is not
        # stable to better than 3x" is the claim the prose stands behind; asserted
        # rather than capped, for the reason given at obf_ratio_floor above.
        "p99_ratio_swing_floor": P99_SWING_FLOOR,
        "p99_rank_from_worst": N_TICKS - int(N_TICKS * 0.99),
        # --- contention, where spin stops looking clever ----------------------
        "sleep_contended_p50_us": stats["sleep_contended"]["p50_us"],
        "spin_contended_p50_us": stats["spin_contended"]["p50_us"],
        "spin_contended_p50_over_sleep": round(
            stats["spin_contended"]["p50_us"] / max(stats["sleep_contended"]["p50_us"], 1e-9), 2
        ),
        "sleep_contended_p99_us": stats["sleep_contended"]["p99_us"],
        "contention_sleep_p99_blowup": round(
            stats["sleep_contended"]["p99_us"] / max(stats["sleep_idle"]["p99_us"], 1e-9), 1
        ),
        # Fraction of ticks over budget with 8 threads competing for the core.
        "sleep_contended_over_budget_frac": stats["sleep_contended"]["over_budget_frac"],
        "spin_contended_over_budget_frac": stats["spin_contended"]["over_budget_frac"],
        # Kept so the run-to-run spread is visible rather than asserted.
        "sleep_idle_p99_us_median_of_passes": median_across_passes("sleep_idle", "p99_us"),
        "spin_idle_p99_us_median_of_passes": median_across_passes("spin_idle", "p99_us"),
        "obf_ratio_median_of_passes": round(
            float(np.median([summarise(p["sleep_idle"])["over_budget_frac"] for p in passes]))
            / max(
                float(np.median([summarise(p["spin_idle"])["over_budget_frac"] for p in passes])),
                1e-9,
            ),
            1,
        ),
    }

    # The floors quoted above are claims; this is where they get checked.  A run
    # that trips one has produced measurements the README is no longer entitled to
    # assert, and the honest outcome is to fail rather than emit the weaker number.
    claims = {
        "obf_ratio_floor": (metrics["obf_ratio"], OBF_RATIO_FLOOR),
        "p50_ratio_floor": (metrics["p50_ratio"], P50_RATIO_FLOOR),
        "p99_ratio_swing_floor": (swing, P99_SWING_FLOOR),
    }
    violated = {k: (got, want) for k, (got, want) in claims.items() if got < want}
    if violated:
        for k, (got, want) in violated.items():
            print(f"  CLAIM VIOLATED: {k} = {got:.2f} < {want} (quoting it would be a lie)")
        return 1

    table = [
        {"pass": i + 1, "condition": c, **summarise(p[c])}
        for i, p in enumerate(passes)
        for c in CONDITIONS
    ]

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"passes": table},
        notes=(
            "Live measurement on a general-purpose OS: volatile by construction, "
            "checked approximately. Headline numbers are pooled across passes and "
            "reproducible; per-pass p99 is not, which is the block's point. "
            "Absolutes and the sleep-vs-spin ordering are host-specific -- on an "
            "isolated core under PREEMPT_RT the ordering inverts."
        ),
    )
    print(
        f"1 kHz loop x{REPEATS} ({metrics['n_samples_pooled']} samples): over-budget "
        f"sleep {metrics['sleep_idle_over_budget_frac']:.2%} vs spin "
        f"{metrics['spin_idle_over_budget_frac']:.2%} (x{metrics['obf_ratio']:.0f}) | "
        f"p50 ratio x{metrics['p50_ratio']:.0f} | contended p50 spin/sleep "
        f"x{metrics['spin_contended_p50_over_sleep']:.2f} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
