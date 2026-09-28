#!/usr/bin/env python
"""Real-time chunking: where the actions come from while the next chunk is still being computed.

03-01 priced staleness.  03-02 built something that emits a whole action chunk in
one shot.  Neither says anything about the **joint** between two chunks, and the
joint is where all the engineering goes.

Three policies, one plant, one error model:

  sync    execute the chunk, then infer the next one.  The arm holds still for one
          forward pass out of every (chunk + forward pass).  Simple, and it has no
          deadline -- the inference may take as long as it likes.
  async   trigger the next inference while the current chunk is still running.  No
          stall, but the fresh chunk is pasted in at the boundary, and its first
          actions were computed from a *different* observation than the actions the
          arm is currently following.  Every boundary steps.
  rtc     same schedule as async, but the new chunk is generated *conditioned* on
          the actions already committed.  Its first P actions are pinned, and past
          that point the new observation takes over -- but not by stepping onto it.
          The conditioned chunk leaves the pinned prefix with matching value and
          slope, and rejoins its own plan over a blend window.

The joint is priced with two numbers, and they pull in opposite directions:

  tracking error   RMS |command - truth| over the episode, in amplitudes.
  discontinuity    the worst second difference of the command, as a multiple of
                   the worst second difference of the truth.  A smooth tracker
                   sits at 1; one that steps at every chunk boundary does not.

The plant is a reference trajectory followed by a perfect servo, exactly as in
03-01: no contact, no friction, no real policy.  The single error mechanism is the
action head's -- every inference draws its own velocity estimate and its own
constant offset, and *that* is why two consecutive chunks disagree at all.

This is a model of the joint, not a reproduction of any published RTC stack.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0

CTRL_HZ = 1000  # servo / command rate
DT = 1.0 / CTRL_HZ
TARGET_F = 1.0  # Hz, same target motion as 03-01
TARGET_A = 1.0  # amplitude; errors are reported as a fraction of this
V_AMP = TARGET_A * 2 * np.pi * TARGET_F  # peak velocity
DURATION_S = 4.0
N_TICKS = int(DURATION_S * CTRL_HZ)

CHUNK_TICKS = 50  # ticks of motion committed per switch -> 20 Hz inference budget
INFER_MS = 20.0  # one forward pass, desktop GPU class (same as 03-01)
LEAD_MS = 20.0  # trigger this early, so the plan is ready exactly on the boundary
LAG_MS = 20.0  # ...and it conditions on a state estimate this old

SIGMA_V = 0.15  # per-inference velocity error, as a fraction of peak velocity
SIGMA_S = 0.02  # per-inference constant offset, as a fraction of amplitude
BLEND_MS = 25.0  # how long the conditioned chunk takes to rejoin its own plan

P_GRID = (0, 10, 20, 30, 40, 50)  # ticks of pinned prefix
BLEND_GRID_MS = (4.0, 8.0, 15.0, 25.0, 40.0, 80.0)

LEAD_TICKS = int(round(LEAD_MS / 1000.0 * CTRL_HZ))
LAG_TICKS = int(round(LAG_MS / 1000.0 * CTRL_HZ))
INFER_TICKS = int(round(INFER_MS / 1000.0 * CTRL_HZ))
# The first index of a chunk the executor is allowed to touch: everything before
# it is already spent.  A prefix shorter than this buys nothing, and that is not
# an assumption, it is what the sweep measures.
INDEX_OFFSET = LEAD_TICKS + LAG_TICKS
P_MIN = INDEX_OFFSET


def pos(t):
    return TARGET_A * np.sin(2 * np.pi * TARGET_F * t)


def vel(t):
    return V_AMP * np.cos(2 * np.pi * TARGET_F * t)


# Every policy replays the *same* noise realisation, so the comparison is between
# schedules and not between draws.  Two draws per inference, drawn up front so the
# stream does not shift when a policy runs a different number of chunks.
NOISE_N = 512
_G = mapkit.rng(SEED)
NOISE_V = _G.standard_normal(NOISE_N)  # velocity-estimate error, one per inference
NOISE_S = _G.standard_normal(NOISE_N)  # constant offset, one per inference


def make_plan(k: int, trigger_t: float, sigma_v: float | None = None, sigma_s: float | None = None):
    """One inference, as a straight-line extrapolation of the state at the anchor.

    The anchor state is exact -- a healthy estimator, so the only error sources are
    the two per-inference draws.  Constant velocity means the plan also carries a
    curvature error that grows quadratically with lookahead, which is the part
    03-01 called the prediction gap.

    Returned as (anchor_s, slope, offset) because every plan here is affine, which
    makes the slope difference between two plans a subtraction -- and makes a plan
    evaluable at any time, which is what lets a prefix be cut out of it.

    ``sigma_v`` / ``sigma_s`` resolve to the block's constants when omitted, so the
    "perfect action head" floor can be run off the same code path.
    """
    sigma_v = SIGMA_V if sigma_v is None else sigma_v
    sigma_s = SIGMA_S if sigma_s is None else sigma_s
    anchor = trigger_t - LAG_MS / 1000.0
    slope = vel(anchor) + sigma_v * V_AMP * NOISE_V[k]
    offset = sigma_s * TARGET_A * NOISE_S[k]
    return (anchor, slope, offset)


def plan_at(plan, t):
    anchor, slope, offset = plan
    return pos(anchor) + slope * (t - anchor) + offset


# --------------------------------------------------------------------------- #
# the three schedules
# --------------------------------------------------------------------------- #


def run_sync(n_ticks: int) -> np.ndarray:
    """Execute a chunk, then infer the next: command is frozen for every forward pass."""
    period = CHUNK_TICKS + INFER_TICKS
    t = np.arange(n_ticks) * DT
    cmd = np.full(n_ticks, pos(-LAG_MS / 1000.0))
    for k in range(int(np.ceil(n_ticks / period))):
        start = k * period
        if start >= n_ticks:
            break
        # the stall: the inference runs here, so the last command is simply held
        hold_end = min(start + INFER_TICKS, n_ticks)
        cmd[start:hold_end] = cmd[start - 1] if k else cmd[start]
        body_end = min(start + period, n_ticks)
        plan = make_plan(k, start * DT)
        cmd[hold_end:body_end] = plan_at(plan, t[hold_end:body_end])
    return cmd


def run_async(n_ticks: int) -> np.ndarray:
    """Trigger early, paste the new chunk in at the boundary.  No stall, but a seam."""
    t = np.arange(n_ticks) * DT
    cmd = np.empty(n_ticks)
    n_chunks = n_ticks // CHUNK_TICKS
    plans = [make_plan(k, (k * CHUNK_TICKS - LEAD_TICKS) * DT) for k in range(n_chunks)]
    for k, plan in enumerate(plans):
        s = slice(k * CHUNK_TICKS, (k + 1) * CHUNK_TICKS)
        cmd[s] = plan_at(plan, t[s])
    return cmd


def run_rtc(n_ticks: int, prefix: int, blend_ms: float = BLEND_MS) -> np.ndarray:
    """Async schedule, but the new chunk is conditioned on the actions already committed.

    The chunk's own timeline is what makes a prefix necessary.  Chunk k is
    triggered LEAD before it is needed and plans forward from an anchor LAG before
    that, so index 0 of the chunk is already LEAD+LAG = 40 ms old by the time the
    executor can touch it.  Those leading actions are not hypothetical: the arm has
    already run through them.  Index ``E = (LEAD + LAG)/dt`` is therefore the first
    index the executor is allowed to use, and the first ``P`` indices are pinned to
    what the previous chunk committed.  Pinning is not a smoothing choice -- it is
    a statement about which actions still exist.

    Two things then have to happen at the freeze boundary:

      continuity  a chunk head trained with inpainting leaves the frozen prefix
                  with matching value *and* slope.  The Hermite-style correction
                  below is that: C1 at the boundary, decaying with length
                  ``blend_ms``, after which the new observation fully owns the
                  chunk.

      cost        the correction is not free.  It is the price of a chunk that
                  disagrees with the stream it is joining, and it is paid in
                  tracking error over the blend window.  Worse, if the boundary
                  sits *before* index E, the blend has partly decayed again by the
                  time the executor arrives -- see ``prefix_sweep``.
    """
    t = np.arange(n_ticks + CHUNK_TICKS) * DT
    cmd = np.full(n_ticks + CHUNK_TICKS, pos(-LAG_MS / 1000.0))
    n_chunks = n_ticks // CHUNK_TICKS + 1
    plans = [make_plan(k, (k * CHUNK_TICKS - LEAD_TICKS) * DT) for k in range(n_chunks)]
    lam = blend_ms / 1000.0
    E = INDEX_OFFSET
    # Hermite basis: value 1 and slope 0 at u=0, both 0 at u=1.
    h1 = lambda u: 2 * u**3 - 3 * u**2 + 1  # noqa: E731
    h2 = lambda u: u**3 - 2 * u**2 + u  # noqa: E731

    for k in range(n_chunks):
        base = k * CHUNK_TICKS
        cur = plans[k]
        if k == 0:
            cmd[base : base + CHUNK_TICKS] = plan_at(cur, t[base : base + CHUNK_TICKS])
            continue

        prev = plans[k - 1]
        rel = base + max(prefix - E, 0)  # first executed tick past the freeze
        bound = base + prefix - E  # tick carrying chunk index P

        # The stream the new chunk has to join.  Behind ``base`` it is committed --
        # already written, never revisable.  At or ahead of it, it is what the
        # previous chunk would have gone on doing.  Value *and slope* have to come
        # from the same one of those two, or the finite difference below straddles
        # a 1 ms window and a 1/DT amplification turns a modelling seam into a
        # runaway.
        behind = bound < base
        c_prev = cmd[bound] if behind else plan_at(prev, t[bound])
        slope_prev = (cmd[bound] - cmd[bound - 1]) / DT if behind else prev[1]

        t0 = t[bound]
        c0 = c_prev - plan_at(cur, t0)
        c1 = slope_prev - cur[1]

        if rel > base:  # prefix reaches into the future: the old plan runs on
            cmd[base:rel] = plan_at(prev, t[base:rel])
        # Finite-support Hermite: starts at (c0, c1) and is back on the new plan,
        # with zero offset and zero slope, after ``lam`` seconds.  Finite support
        # matters -- a purely exponential blend never reaches zero, and the
        # leftover at the end of every chunk is itself a step, which is the thing
        # this whole block is trying to remove.
        s = t[rel : base + CHUNK_TICKS] - t0
        u = np.clip(s / lam, 0.0, 1.0)
        corr = c0 * h1(u) + c1 * lam * h2(u)
        cmd[rel : base + CHUNK_TICKS] = plan_at(cur, t[rel : base + CHUNK_TICKS]) + corr
    return cmd[:n_ticks]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score(cmd: np.ndarray, n_ticks: int) -> dict:
    t = np.arange(n_ticks) * DT
    truth = pos(t)
    err = cmd - truth
    d2c = np.diff(cmd, 2)
    d2t = np.diff(truth, 2)
    return {
        "rms_pct": round(100.0 * float(np.sqrt((err**2).mean())) / TARGET_A, 4),
        "peak_jerk": round(float(np.abs(d2c).max() / np.abs(d2t).max()), 2),
        "rms_jerk": round(float(np.sqrt((d2c**2).mean() / (d2t**2).mean())), 2),
    }


def main() -> int:
    rows = [
        # A perfect action head on the async schedule: what the curvature of the
        # target motion costs on its own, before any inference error exists.
        dict(policy="floor", **score(_run_async_clean(), N_TICKS)),
        dict(policy="sync", **score(run_sync(N_TICKS), N_TICKS)),
        dict(policy="async", **score(run_async(N_TICKS), N_TICKS)),
        dict(policy="rtc", **score(run_rtc(N_TICKS, P_MIN), N_TICKS)),
    ]
    by = {r["policy"]: r for r in rows}

    p_rows = []
    for p in P_GRID:
        r = score(run_rtc(N_TICKS, p), N_TICKS)
        p_rows.append(
            {
                "prefix_ticks": p,
                "prefix_ms": round(p * DT * 1000.0, 2),
                "past_pct": round(100.0 * p / INDEX_OFFSET, 1),
                # below INDEX_OFFSET the whole blend happens in the past, where the
                # executor cannot see it -- the row is then byte-identical to async
                "conditions_live_region": p >= INDEX_OFFSET,
                **r,
            }
        )
    p_by = {r["prefix_ticks"]: r for r in p_rows}
    p_dead = max(p for p in P_GRID if p < INDEX_OFFSET)

    b_rows = []
    for b in BLEND_GRID_MS:
        r = score(run_rtc(N_TICKS, P_MIN, blend_ms=b), N_TICKS)
        b_rows.append({"blend_ms": b, "fits_in_chunk": b <= CHUNK_TICKS * DT * 1000.0, **r})
    b_ok = [r for r in b_rows if r["fits_in_chunk"]]
    b_by = {r["blend_ms"]: r for r in b_rows}

    metrics = {
        "ctrl_hz": CTRL_HZ,
        "target_f_hz": TARGET_F,
        "target_a": TARGET_A,
        "duration_s": DURATION_S,
        "chunk_ticks": CHUNK_TICKS,
        "chunk_ms": round(CHUNK_TICKS * DT * 1000.0, 2),
        "trigger_hz": round(CTRL_HZ / CHUNK_TICKS, 2),
        "infer_ms": INFER_MS,
        "lead_ms": LEAD_MS,
        "lag_ms": LAG_MS,
        "lead_ticks": LEAD_TICKS,
        "lag_ticks": LAG_TICKS,
        "infer_ticks": INFER_TICKS,
        # The whole reason a prefix exists: index 0 of a chunk is this old by the
        # time the executor may touch it.
        "index_offset": INDEX_OFFSET,
        "index_offset_ms": round(INDEX_OFFSET * DT * 1000.0, 2),
        "sigma_v": SIGMA_V,
        "sigma_s": SIGMA_S,
        "blend_ms": BLEND_MS,
        # sync spends one forward pass out of every period holding still.
        "sync_period_ms": round((CHUNK_TICKS + INFER_TICKS) * DT * 1000.0, 2),
        "sync_stall_pct": round(100.0 * INFER_TICKS / (CHUNK_TICKS + INFER_TICKS), 2),
        # headline comparison
        "floor_rms_pct": by["floor"]["rms_pct"],
        "floor_peak_jerk": by["floor"]["peak_jerk"],
        "sync_rms_pct": by["sync"]["rms_pct"],
        "sync_peak_jerk": by["sync"]["peak_jerk"],
        "async_rms_pct": by["async"]["rms_pct"],
        "async_peak_jerk": by["async"]["peak_jerk"],
        "async_rms_jerk": by["async"]["rms_jerk"],
        "rtc_rms_pct": by["rtc"]["rms_pct"],
        "rtc_peak_jerk": by["rtc"]["peak_jerk"],
        "rtc_rms_jerk": by["rtc"]["rms_jerk"],
        "jerk_cut": round(by["async"]["peak_jerk"] / by["rtc"]["peak_jerk"], 2),
        "rtc_track_cost": round(by["rtc"]["rms_pct"] / by["async"]["rms_pct"], 3),
        # the prefix sweep
        "p_min": P_MIN,
        "p_min_ms": round(P_MIN * DT * 1000.0, 2),
        "p_dead_max": p_dead,
        "rms_at_p_dead": p_by[p_dead]["rms_pct"],
        "jerk_at_p_dead": p_by[p_dead]["peak_jerk"],
        "rms_at_p_max": p_by[max(P_GRID)]["rms_pct"],
        "jerk_at_p_max": p_by[max(P_GRID)]["peak_jerk"],
        "prefix_track_cost": round(p_by[max(P_GRID)]["rms_pct"] / p_by[P_MIN]["rms_pct"], 3),
        # the blend sweep, restricted to windows that fit inside a chunk
        "blend_short_ms": min(r["blend_ms"] for r in b_ok),
        "blend_short_jerk": min(b_ok, key=lambda r: r["blend_ms"])["peak_jerk"],
        "blend_long_ms": max(r["blend_ms"] for r in b_ok),
        "blend_long_jerk": max(b_ok, key=lambda r: r["blend_ms"])["peak_jerk"],
        "blend_overlong_ms": max(BLEND_GRID_MS),
        "blend_overlong_jerk": b_by[max(BLEND_GRID_MS)]["peak_jerk"],
        "blend_shortest_rms": min(b_ok, key=lambda r: r["blend_ms"])["rms_pct"],
        "blend_longest_rms": max(b_ok, key=lambda r: r["blend_ms"])["rms_pct"],
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"policies": rows, "prefix_sweep": p_rows, "blend_sweep": b_rows},
    )
    print(
        f"jerk async {by['async']['peak_jerk']:.0f} -> rtc {by['rtc']['peak_jerk']:.0f} "
        f"({metrics['jerk_cut']:.0f}x) | track {metrics['async_rms_pct']:.2f}% -> "
        f"{metrics['rtc_rms_pct']:.2f}% ({metrics['rtc_track_cost']:.2f}x) -> {path.name}"
    )
    return 0


def _run_async_clean() -> np.ndarray:
    """Async schedule with a perfect action head, i.e. the curvature-only floor."""
    t = np.arange(N_TICKS) * DT
    cmd = np.empty(N_TICKS)
    for k in range(N_TICKS // CHUNK_TICKS):
        plan = make_plan(k, (k * CHUNK_TICKS - LEAD_TICKS) * DT, 0.0, 0.0)
        s = slice(k * CHUNK_TICKS, (k + 1) * CHUNK_TICKS)
        cmd[s] = plan_at(plan, t[s])
    return cmd


if __name__ == "__main__":
    raise SystemExit(main())
