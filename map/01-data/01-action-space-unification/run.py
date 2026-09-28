#!/usr/bin/env python
"""Unified action space, priced as the cost of *not* having one.

Three heterogeneous arms solve the same semantic task -- move the end effector to the
observed relative target (dx, dy, dz) -- but each speaks its own action language:
radians, normalised counts, metres mixed with centimetres.  We fit one linear policy
``observation -> action`` on robot A and test it on robot B, twice:

  raw       A's joint commands in A's own units, zero-padded to the widest robot.
  unified   every robot min-max normalised to [-1, 1] and zero-padded, carrying a
            validity mask so padded dims contribute nothing to the fit.

Both errors are scored in the target's raw action units, on the arm-joint dims the two
robots share.  The gap is the price of a per-robot action convention, and nothing else.

The second experiment prices the same mistake for an autoregressive token head:
per-robot bins mean per-robot vocabulary, and the vocabulary stops being a property of
the *task* and becomes a property of the *body count*.

Ground truth is built so that, *after* per-robot normalisation, all three robots share
one operator up to a small per-robot idiosyncrasy.  That is exactly the assumption
unification buys; the block measures what happens when you refuse to buy it.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0
N = 4000  # (observation, action) pairs per robot
BINS = 256  # discreteness of one action dimension in the token head
OBS_HALF = 0.6  # relative target drawn from U(-OBS_HALF, OBS_HALF), metres
DELTA = 0.08  # per-robot kinematic idiosyncrasy, in normalised units
NOISE = 0.02  # action noise, in normalised units
NORM_RANGE = (-1.0, 1.0)  # the shared interval after min-max
DELTA_SWEEP = (0.0, 0.02, 0.05, 0.08, 0.15, 0.30)

# name, arm joints, action unit, per-joint (lo, hi) in that unit, gripper (lo, hi)
ROBOTS = (
    ("A", 6, "弧度 rad", ((-3.14, 3.14),) * 6, (0.0, 1.0)),
    ("B", 7, "归一化", ((-1.0, 1.0),) * 7, (0.0, 1000.0)),
    ("C", 3, "米 / 厘米混用", ((-0.3, 0.3), (-20.0, 20.0), (-0.1, 0.1)), (0.0, 100.0)),
)


def to_raw(r: dict, u: np.ndarray) -> np.ndarray:
    """Map a normalised action in NORM_RANGE back onto the robot's own raw units."""
    span = NORM_RANGE[1] - NORM_RANGE[0]
    return r["lo"] + (u - NORM_RANGE[0]) / span * (r["hi"] - r["lo"])


def build() -> list[dict]:
    """One record per robot.  The gripper always sits last, so the shared arm-joint
    prefix is index-aligned across bodies and the gripper is index-misaligned."""
    out = []
    for name, arm, unit, joints, grip in ROBOTS:
        ranges = list(joints) + [grip]
        lo = np.array([r[0] for r in ranges], dtype=float)
        hi = np.array([r[1] for r in ranges], dtype=float)
        out.append(
            {
                "name": name,
                "arm": arm,
                "unit": unit,
                "ranges": ranges,
                "lo": lo,
                "hi": hi,
                "dof": len(ranges),
            }
        )
    return out


def dataset(rs: list[dict], K: np.ndarray, delta: float, obs: np.ndarray):
    """(u, a) per robot: the normalised action and the same action in raw units.

    Each robot has its own linear operator ``K[:dof] + delta * direction`` -- the
    "everyone's kinematics differ" part -- expressed in its own unit scale."""
    out = {}
    for i, r in enumerate(rs):
        d = r["dof"]
        dirn = mapkit.rng(SEED + 7 * i).normal(0.0, 1.0, (d, 3))
        dirn /= np.sqrt((dirn**2).sum(axis=1, keepdims=True))
        u = obs @ (K[:d] + delta * dirn).T
        u += mapkit.rng(SEED + 1000 + 7 * i).normal(0.0, NOISE, u.shape)
        out[r["name"]] = (u, to_raw(r, u))
    return out


def at(rs: list[dict], name: str) -> dict:
    return next(r for r in rs if r["name"] == name)


def transfer(rs, data, obs, src: str, dst: str, pad: int, unified: bool) -> float:
    """Fit on ``src``, score on ``dst`` -- in dst's raw units, on shared arm joints."""
    S, D = at(rs, src), at(rs, dst)
    # The gripper sits last in every body, so min(dof) - 1 compares index-aligned arm
    # joints only.  Gripper alignment is a separate problem, excluded on purpose.
    dims = np.arange(min(S["dof"], D["dof"]) - 1)
    mask = np.zeros(pad, dtype=bool)
    mask[: S["dof"]] = True
    y = np.zeros((len(obs), pad))
    y[:, : S["dof"]] = data[src][0 if unified else 1]

    # Least squares per output dim -- but only on dims the source robot actually has,
    # which is what the validity mask is for.  Padded dims keep their zero column.
    X = np.hstack([obs, np.ones((len(obs), 1))])
    W = np.zeros((X.shape[1], pad))
    for d in np.flatnonzero(mask):
        W[:, d] = np.linalg.lstsq(X, y[:, d], rcond=None)[0]

    pred = X @ W
    if unified:  # back to the target's raw units before scoring
        lo, hi = np.zeros(pad), np.zeros(pad)
        lo[: D["dof"]], hi[: D["dof"]] = D["lo"], D["hi"]
        pred = to_raw({"lo": lo, "hi": hi}, pred)
    return float(np.abs(pred[:, dims] - data[dst][1][:, dims]).mean())


def main() -> int:
    rs = build()
    pad = max(r["dof"] for r in rs)
    dof_sum = sum(r["dof"] for r in rs)

    gen = mapkit.rng(SEED)
    K = gen.uniform(0.3, 0.7, (pad, 3)) * gen.choice([-1.0, 1.0], (pad, 3))
    obs = gen.uniform(-OBS_HALF, OBS_HALF, (N, 3))
    data = dataset(rs, K, DELTA, obs)

    pairs = [(s, d) for s in ("A", "B", "C") for d in ("A", "B", "C") if s != d]
    rows = []
    for src, dst in pairs:
        raw = transfer(rs, data, obs, src, dst, pad, unified=False)
        uni = transfer(rs, data, obs, src, dst, pad, unified=True)
        rows.append(
            {
                "from": src,
                "to": dst,
                "common_dims": min(at(rs, src)["dof"], at(rs, dst)["dof"]) - 1,
                "raw_mae": round(raw, 4),
                "unified_mae": round(uni, 4),
                "reduction": round(raw / uni, 2),
            }
        )
    main_row = next(r for r in rows if r["from"] == "A" and r["to"] == "B")

    sweep = []
    for delta in DELTA_SWEEP:
        d_ = dataset(rs, K, delta, obs)
        raw = transfer(rs, d_, obs, "A", "B", pad, unified=False)
        uni = transfer(rs, d_, obs, "A", "B", pad, unified=True)
        sweep.append(
            {
                "delta": delta,
                "raw_mae": round(raw, 4),
                "unified_mae": round(uni, 4),
                "reduction": round(raw / uni, 2),
            }
        )

    # Token head: per-body bins vs one shared set of bins.
    vocab_raw = dof_sum * BINS
    vocab_unified = pad * BINS

    metrics = {
        "n_robots": len(rs),
        "n_samples_per_robot": N,
        "dof_sum": dof_sum,
        "dof_max": pad,
        "norm_range": list(NORM_RANGE),
        "obs_half_range_m": OBS_HALF,
        "action_noise_sigma": NOISE,
        "kinematic_idiosyncrasy": DELTA,
        "transfer_mae_raw": main_row["raw_mae"],
        "transfer_mae_unified": main_row["unified_mae"],
        "transfer_error_reduction": main_row["reduction"],
        "bins_per_dim": BINS,
        "vocab_raw": vocab_raw,
        "vocab_unified": vocab_unified,
        "vocab_compression": round(vocab_raw / vocab_unified, 3),
        "vocab_per_robot_raw": [r["dof"] * BINS for r in rs],
    }

    tables = {
        "robots": [
            {
                "robot": r["name"],
                "arm_joints": r["arm"],
                "dof": r["dof"],
                "unit": r["unit"],
                "joint_lo": [x[0] for x in r["ranges"][:-1]],
                "joint_hi": [x[1] for x in r["ranges"][:-1]],
                "gripper_lo": r["ranges"][-1][0],
                "gripper_hi": r["ranges"][-1][1],
                "vocab_raw": r["dof"] * BINS,
            }
            for r in rs
        ],
        "transfer": rows,
        "idiosyncrasy_sweep": sweep,
    }

    path = mapkit.emit(pathlib.Path(__file__).resolve().parent, metrics, tables, seed=SEED)
    print(
        f"A->B MAE raw {main_row['raw_mae']:.4f} -> unified {main_row['unified_mae']:.4f} "
        f"({main_row['reduction']}x) | vocab {vocab_raw} -> {vocab_unified} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
