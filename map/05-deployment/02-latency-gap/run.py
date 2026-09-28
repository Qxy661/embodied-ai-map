#!/usr/bin/env python
"""Two-rate hierarchy: the window in which a slow brain is allowed to be slow.

A single network cannot both think and twitch.  One forward pass of a VLA-sized
model costs SLOW_LAT_MS; the plant's control loop wants a 1 kHz tick.  This block
makes that incompatibility arithmetic instead of rhetorical, and then asks what
the industry's answer -- split the loop in two -- actually buys.

The model is a *delay budget* and nothing more:

    the fast brain acts on an intent that is L seconds old (the slow brain's own
    compute) plus up to a whole hold period old (it learns the next intent only
    when the slow brain finishes it), and a pure delay t turns a sinusoid of
    frequency f into a tracking error of 2*sin(pi*f*t) amplitudes.

Two fast-brain policies bracket the design space, mirroring 03-decision/01:

  hold    zero-order hold on the latest intent -- a buffer that repeats itself.
  extrap  first-order hold: measure the last slope, carry it forward.  An upper
          bound on what a learned low-level policy could recover from the same
          intent stream.

A real fast brain lands between them.  The gap is what a predictive low-level
policy is worth, priced in *how slow the slow brain is allowed to be*.

Two-sided constraint, and this is the whole point:

    f_slow >= f_min_track   (below it the fast brain cannot fill the gap)
    f_slow <= 1 / L         (above it the slow brain cannot finish a pass in time)

A single-rate policy has no such window to stand in: it must close the loop at
the plant's rate *and* respect its own latency, and those two numbers are three
decades apart.

This is a mechanism model, not a policy reproduction -- no plant, no closed loop,
no learned weights.  It can show that *under these two constraints* a single-rate
policy has no solution.  It cannot show that a real robot must be hierarchical.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

F_FAST = 1000.0  # the tick the plant's control loop is closed at
F_TASK = 0.5  # bandwidth of the motion the task demands
AMP = 1.0  # amplitude; every error below is a fraction of this
SLOW_LAT_MS = 30.0  # one slow-brain forward pass, in milliseconds
ERR_BUDGET = 0.30  # RMS tracking error we are still willing to call "working"
DURATION_S = 10.0
WARMUP_S = 2.0  # discarded: the first publish has no slope to extrapolate from
MAX_STEP = 2000  # slowest slow brain we consider: 0.5 Hz

# The sweep runs on the slow brain's *period in control ticks*, not on a nominal
# frequency.  Quantising to ticks is not pedantry -- at the ceiling the period is
# exactly SLOW_LAT_MS ticks, and a rate that is not representable as a whole
# number of ticks is not a rate the loop can actually run at.
F_SLOW_SWEEP = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0]
TASK_SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def _ref(t, f_task: float):
    return AMP * np.sin(2.0 * np.pi * f_task * t)


def rms_error(step: int, mode: str, f_task: float = F_TASK) -> float:
    """RMS |command - reference| for a slow brain that publishes every `step` ticks.

    Fully vectorised: the publish instants are exactly periodic in tick space
    (``nxt = t[k] + period`` makes the gap a constant number of ticks), so the
    whole episode is one searchsorted plus one gather.
    """
    dt = 1.0 / F_FAST
    n = int(round(DURATION_S * F_FAST))
    ticks = np.arange(n)
    t = ticks * dt

    pub = np.arange(0, n, step)
    t_pub = pub * dt
    # the intent the slow brain publishes was computed from an L-old observation
    intent = _ref(t_pub - SLOW_LAT_MS / 1000.0, f_task)

    at = np.searchsorted(pub, ticks, side="right") - 1
    cmd = intent[at]
    if mode == "extrap":
        # slope measured between the last two published intents, carried forward
        slope = np.zeros_like(intent)
        slope[1:] = np.diff(intent) / np.diff(t_pub)
        cmd = cmd + slope[at] * (t - t_pub[at])

    w = int(round(WARMUP_S * F_FAST))
    return float(np.sqrt(((cmd[w:] - _ref(t, f_task)[w:]) ** 2).mean()))


def zoh_rms_analytic(f_slow: float, f_task: float = F_TASK) -> float:
    """Closed form for `hold`, as an independent check on the simulation.

    Exactly the RMS of ``r(t_i - L) - r(t)`` for a sinusoid, assuming the hold
    phase averages uniformly.  That assumption holds when the hold period is
    incommensurate with the motion period and degrades once it is not -- which is
    why this is only used on the high-rate end of the sweep (see n_analytic).
    """
    w = 2.0 * np.pi * f_task
    lat = SLOW_LAT_MS / 1000.0
    per = 1.0 / f_slow
    val = 1.0 - (np.sin(w * (lat + per)) - np.sin(w * lat)) / (w * per)
    return float(np.sqrt(max(val, 0.0)))


def slowest_step(mode: str, f_task: float = F_TASK, cap: int = MAX_STEP) -> int | None:
    """Longest hold period, in ticks, whose RMS error still meets the budget.

    Integer bisection, so the answer is a rate the loop can really run at and the
    boundary is tight by construction: `slowest_step + 1` tick misses the budget.

    ``None`` means *no* rate clears the budget -- the pure L-second latency of the
    slow brain already puts the task out of reach, before any hold period is
    added.  This is the case where the architecture has no window at all, and it
    is a result, not an error.
    """
    if rms_error(1, mode, f_task) > ERR_BUDGET:
        return None
    if rms_error(cap, mode, f_task) <= ERR_BUDGET:
        return cap  # no lower bound binds within the range we sweep
    lo, hi = 1, cap  # lo meets budget, hi does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rms_error(mid, mode, f_task) <= ERR_BUDGET:
            lo = mid
        else:
            hi = mid
    return lo


def task_ceiling(mode: str, ceiling_ticks: int) -> float:
    """Largest motion bandwidth a slow brain this slow can still serve.

    Above it the delay budget is exhausted before the compute ceiling is even
    reached, and the window closes: no slow rate satisfies both sides.
    """

    def ok(f_task: float) -> bool:
        s = slowest_step(mode, f_task)
        return s is not None and s >= ceiling_ticks

    lo, hi = TASK_SWEEP[0], 20.0  # lo has a window, hi does not
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main() -> int:
    ceiling_step = int(np.ceil(SLOW_LAT_MS / 1000.0 * F_FAST))
    f_max = F_FAST / SLOW_LAT_MS

    rows = []
    for nominal in F_SLOW_SWEEP:
        step = int(np.ceil(F_FAST / nominal))
        realized = F_FAST / step
        e_hold = rms_error(step, "hold")
        e_extrap = rms_error(step, "extrap")
        compute_ok = realized <= f_max
        rows.append(
            {
                "f_slow_nominal_hz": nominal,
                "period_ms": round(1000.0 / realized, 3),
                "f_slow_realized_hz": round(realized, 3),
                "rms_hold": round(e_hold, 6),
                "rms_extrap": round(e_extrap, 6),
                "compute_ok": bool(compute_ok),
                "track_ok_hold": bool(e_hold <= ERR_BUDGET),
                "track_ok_extrap": bool(e_extrap <= ERR_BUDGET),
                "feasible_hold": bool(compute_ok and e_hold <= ERR_BUDGET),
                "feasible_extrap": bool(compute_ok and e_extrap <= ERR_BUDGET),
            }
        )

    # The two sides are both statements about the slow brain's *period*, which is
    # why they are compared in ticks.  Tracking wants the period short (a small
    # step); the compute ceiling wants the period at least as long as one forward
    # pass (a step of at least `ceiling_step` ticks).  The window is the overlap.
    step_hold = slowest_step("hold")
    step_extrap = slowest_step("extrap")
    f_min_hold = F_FAST / step_hold
    f_min_extrap = F_FAST / step_extrap
    ceil_hold = task_ceiling("hold", ceiling_step)
    ceil_extrap = task_ceiling("extrap", ceiling_step)

    task_rows = []
    for f_task in TASK_SWEEP:
        s_h = slowest_step("hold", f_task)
        s_e = slowest_step("extrap", f_task)
        task_rows.append(
            {
                "f_task_hz": f_task,
                # "none" = no slow rate clears the budget at this bandwidth
                "f_min_hold_hz": "none" if s_h is None else round(F_FAST / s_h, 3),
                "f_min_extrap_hz": "none" if s_e is None else round(F_FAST / s_e, 3),
                "rms_floor_hold": round(rms_error(1, "hold", f_task), 6),
                "rms_floor_extrap": round(rms_error(1, "extrap", f_task), 6),
                # a slower slow brain needs a *longer* period than its own
                # latency, i.e. step >= ceiling_step; see the note in main()
                "window_hold_ok": bool(s_h is not None and s_h >= ceiling_step),
                "window_extrap_ok": bool(s_e is not None and s_e >= ceiling_step),
            }
        )

    # The bisection trusts that the rms curve is monotone in the hold period.  It
    # is, apart from a small beat between the hold period and the motion period.
    # Quantifying the residual ripple is the error bar on the bisected bound.
    def ripple(mode: str, hi: int = 600) -> float:
        e = np.array([rms_error(s, mode) for s in range(1, hi + 1)])
        return float(np.diff(e).max())

    # Independent check: the simulation must reproduce the closed form for `hold`
    # wherever the hold phase averages cleanly (motion phase advance per hold
    # period <= 1 rad).  Outside that the closed form is the thing that is wrong,
    # not the simulation, so those points are excluded by construction.
    devs = [
        abs(rms_error(int(np.ceil(F_FAST / f)), "hold") - zoh_rms_analytic(f))
        for f in F_SLOW_SWEEP
        if 2.0 * np.pi * F_TASK / f <= 1.0
    ]

    metrics = {
        "f_fast_hz": F_FAST,
        "f_task_hz": F_TASK,
        "amp": AMP,
        "slow_lat_ms": SLOW_LAT_MS,
        "err_budget_rms": ERR_BUDGET,
        "duration_s": DURATION_S,
        "warmup_s": WARMUP_S,
        # --- the two-sided constraint on the slow rate -----------------------
        "f_max_compute_hz": round(f_max, 3),
        "ceiling_step_ticks": ceiling_step,
        "f_min_track_hold_hz": round(f_min_hold, 3),
        "f_min_track_extrap_hz": round(f_min_extrap, 3),
        "window_hold_ok": bool(step_hold >= ceiling_step),
        "window_extrap_ok": bool(step_extrap >= ceiling_step),
        "window_hold_ratio": round(f_max / f_min_hold, 3),
        "window_extrap_ratio": round(f_max / f_min_extrap, 3),
        # the boundary shown tight: step_hold meets the budget, +1 tick misses it
        "rms_hold_at_f_min": round(rms_error(step_hold, "hold"), 6),
        "rms_hold_below_f_min": round(rms_error(step_hold + 1, "hold"), 6),
        # --- what the fast brain has to invent between two intents ----------
        "cycles_per_intent_at_f_min": int(step_hold),
        "cycles_per_intent_at_ceiling": int(ceiling_step),
        # --- the error floor, and what the ceiling leaves on the table ------
        "rms_hold_floor": round(rms_error(1, "hold"), 6),
        "rms_hold_at_ceiling": round(rms_error(ceiling_step, "hold"), 6),
        "rms_hold_at_1hz": round(rms_error(1000, "hold"), 6),
        # --- what a predictive fast brain buys ------------------------------
        "slowdown_gain_extrap": round(step_extrap / step_hold, 3),
        # --- where the window closes entirely -------------------------------
        "task_ceiling_hold_hz": round(ceil_hold, 3),
        "task_ceiling_extrap_hz": round(ceil_extrap, 3),
        "task_ceiling_gain": round(ceil_extrap / ceil_hold, 3),
        # --- the single-rate alternative ------------------------------------
        "single_layer_required_hz": F_FAST,
        "single_layer_available_hz": round(f_max, 3),
        "single_layer_shortfall_x": round(F_FAST / f_max, 1),
        # --- cross-check -----------------------------------------------------
        "n_analytic_points": len(devs),
        "analytic_max_dev": round(max(devs), 6),
        "ripple_max_hold": round(ripple("hold"), 6),
        "ripple_max_extrap": round(ripple("extrap"), 6),
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent, metrics, {"sweep": rows, "task_sweep": task_rows}
    )
    print(
        f"slow-brain window {f_min_hold:.2f}-{f_max:.2f} Hz "
        f"({f_max / f_min_hold:.2f}x wide, {step_hold} control cycles per intent); "
        f"extrap opens it to {f_max / f_min_extrap:.2f}x; "
        f"single-rate needs {F_FAST / f_max:.0f}x the rate it has -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
