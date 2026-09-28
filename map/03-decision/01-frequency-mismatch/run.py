#!/usr/bin/env python
"""Action chunking, measured as a pure staleness problem.

The plant is deliberately trivial: we assume the low-level servo tracks the command
perfectly and measure only **how stale the command is**, in units of the target
amplitude.  That isolates the mechanism we care about -- the 30 Hz observation /
1 kHz control mismatch -- from every other source of tracking error.

Two policies bracket the design space:

  reactive   commands the last observed position and holds it for the whole chunk.
             This is what you get if the policy has no model of the future.
  oracle     extrapolates using the true velocity.  An upper bound on what any
             learned predictive policy (flow matching, diffusion, ...) could do.

A real VLA lands between the two.  The gap between them is the entire reason action
chunking requires a *generative* action head rather than a reactive regressor.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

CTRL_HZ = 1000  # joint servo rate
OBS_HZ = 30  # camera / VLM observation rate
INFER_MS = 20.0  # one forward pass, desktop GPU class
MAX_INFER_HZ = 20.0  # what a policy server can sustain end to end
DURATION_S = 4.0
TARGET_F = 1.0  # Hz of the target motion
TARGET_A = 1.0  # amplitude; errors are reported as a fraction of this
CHUNKS = [1, 5, 10, 25, 50, 100]


def target(t):
    return TARGET_A * np.sin(2 * np.pi * TARGET_F * t)


def target_vel(t):
    return TARGET_A * 2 * np.pi * TARGET_F * np.cos(2 * np.pi * TARGET_F * t)


def episode(h: int, mode: str) -> tuple[float, float]:
    """Mean and RMS |command - target| over one episode, in amplitude units."""
    dt = 1.0 / CTRL_HZ
    n = int(DURATION_S * CTRL_HZ)
    t = np.arange(n) * dt
    truth = target(t)
    cmd = np.empty(n)

    for k in range(0, n, h):
        # The observation the policy acts on is already INFER_MS old when the
        # chunk starts executing -- this is the part that does not go away.
        t_obs = t[k] - INFER_MS / 1000.0
        p, v = target(t_obs), target_vel(t_obs)
        seg = slice(k, min(k + h, n))
        tau = t[seg] - t[k]
        cmd[seg] = p + v * tau if mode == "oracle" else p

    err = np.abs(cmd - truth)
    return float(err.mean()), float(np.sqrt((err**2).mean()))


def main() -> int:
    rows = []
    for h in CHUNKS:
        need_hz = CTRL_HZ / h
        r_mae, r_rms = episode(h, "reactive")
        o_mae, o_rms = episode(h, "oracle")
        rows.append(
            {
                "H": h,
                "chunk_ms": round(1000.0 * h / CTRL_HZ, 2),
                "required_infer_hz": round(need_hz, 2),
                "reactive_mae": round(r_mae, 6),
                "oracle_mae": round(o_mae, 6),
                "reactive_rms": round(r_rms, 6),
                "oracle_rms": round(o_rms, 6),
                "feasible": need_hz <= MAX_INFER_HZ,
            }
        )

    by_h = {r["H"]: r for r in rows}
    h_min = min(r["H"] for r in rows if r["feasible"])

    # How much of the error a predictive head removes, at the chunk length the
    # feasibility bound forces you to use.
    r50, o50 = by_h[50]["reactive_mae"], by_h[50]["oracle_mae"]

    metrics = {
        "ctrl_hz": CTRL_HZ,
        "obs_hz": OBS_HZ,
        "ctrl_obs_ratio": round(CTRL_HZ / OBS_HZ, 3),
        "target_f_hz": TARGET_F,
        "target_a": TARGET_A,
        "duration_s": DURATION_S,
        # The irreducible cost of inference latency, before any policy exists.
        "latency_floor_pct": round(by_h[1]["reactive_mae"] * 100, 3),
        "infer_ms": INFER_MS,
        "max_infer_hz": MAX_INFER_HZ,
        "h_min_feasible": h_min,
        "h1_required_infer_hz": by_h[1]["required_infer_hz"],
        "reactive_mae_at_H50": r50,
        "oracle_mae_at_H50": o50,
        "prediction_gain_at_H50": round(r50 / o50, 3),
        "reactive_mae_at_H1": by_h[1]["reactive_mae"],
        "oracle_mae_at_H1": by_h[1]["oracle_mae"],
        # Growth from no-chunking (H=1) to the feasibility bound (H=50).  The
        # whole argument for a generative action head lives in the gap between
        # these two numbers.
        "reactive_growth_H1_to_H50": round(by_h[50]["reactive_mae"] / by_h[1]["reactive_mae"], 3),
        "oracle_growth_H1_to_H50": round(by_h[50]["oracle_mae"] / by_h[1]["oracle_mae"], 3),
    }

    path = mapkit.emit(pathlib.Path(__file__).resolve().parent, metrics, {"sweep": rows})
    print(f"H>={h_min} feasible | err@H50 reactive {r50:.4f} vs oracle {o50:.4f} -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
