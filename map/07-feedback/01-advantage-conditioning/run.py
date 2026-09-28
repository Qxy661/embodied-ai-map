#!/usr/bin/env python
"""Not every demonstration is worth the same, and the difference is only visible
off the training distribution.

Robot data is the binding constraint of the field: a real trajectory costs an
operator, a rig and a room, and there is no fleet on the road collecting it for
free.  So the question is not "how do we get more data" but "how do we make each
trajectory worth more" -- and the naive answer, treating every demonstration as
one equally weighted maximum-likelihood sample, is the thing this block measures
against.

The model is deliberately small enough to read end to end:

  * A **scene axis** ``s in [0, 1]``, which is also the difficulty.  The expert
    action is ``a*(s) = sin(pi s)``.  The pool is collected where collection is
    cheap, so scene density falls off towards the hard end.

  * A **difficulty ceiling**.  Below it the probability of success falls off
    linearly; above it every single attempt fails.  That is the one structural
    assumption this block rests on, and it is the reason failures are not
    worthless: past the ceiling, a failed run is the *only* witness to what the
    scene looks like.

  * A **latent quality** ``q``: 1 is a clean expert run, 0 is flailing.  A
    successful run is never as bad as a failed one, so ``q`` is drawn from a high
    band on success and a low band on failure.  The executed actions deviate from
    ``a*(s)`` by ``(1 - q)`` times a bias plus noise -- a flailing run is both
    systematically off and jittery.

  * A **critic** that scores each demonstration.  Crucially it scores the
    *estimated* advantage ``A_hat = A + sigma * eta``, not the true one.  Any
    claim about weighting that assumes a perfect value function is not a claim
    about the real system, so the block also reports what the same filter would
    score if the outcome were known exactly.

The policy is a weighted Nadaraya-Watson kernel smoother over ``s``: the action
at a scene is the weight-averaged action of the demonstrations near it.  That is
the entire learner -- one line of numpy, no fitting, deterministic.  It is chosen
because it makes the two failure modes *visible*: bad demonstrations drag the
average (bias), and filtered-out demonstrations leave a hole the kernel fills
from somewhere far away (coverage).

Four ways to use the identical pool, differing only in the weight vector:

  uniform    w = 1                            plain behaviour cloning
  filtered   w = 1[critic says success]       keep only the runs that worked
  awr        w = exp(A_hat / beta)            advantage-weighted regression
  awr+floor  w = max(exp(A_hat/beta), eps)    ...but never drop a demo entirely

and two sweeps over the knob that matters: ``beta`` (how hard the weighting
bites) and ``eps`` (how much of the bad data is bought back).

Everything is averaged over replicas.  A single pool of 200 demonstrations is one
draw, and the aggressive policies are *high variance* -- which is precisely the
part a single draw would hide.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0
N_REPLICAS = 40
N_DEMOS = 200

# Pool scene density is proportional to (1 - s) ** SCENE_SKEW: data is collected
# where collection is cheap and the robot already works.
SCENE_SKEW = 1.0

# Above this scene coordinate, no attempt ever succeeds.  Below it, the success
# rate falls off linearly to zero.  This is the block's one structural claim.
SUCCESS_CEILING = 0.75
P_SUCCESS_FLOOR = 0.0  # no rescue above the ceiling

# Latent quality bands.  A success lives in [0.75, 1], a failure in [0, 0.15] --
# the bands are disjoint on purpose, so "filtered BC" is exactly "keep successes".
Q_SUCCESS_LO = 0.75
Q_FAIL_HI = 0.15

# How far a flailing demonstration lands from the expert action.  Small relative
# to the scene gradient, which is what gives a failure in a data-starved region
# any value at all.
FAIL_BIAS = 0.35
FAIL_NOISE = 0.15

# Std of the critic's advantage error.  Zero here would be the dishonest version
# of this block: it would say "weighting is free".
ADV_NOISE = 0.4

# Kernel smoother bandwidth, in scene units.  Fixed across every policy; this
# block compares weightings, not bandwidths.
BANDWIDTH = 0.06

# Training RMSE is density-weighted by resampling the grid onto the pool's own
# scene quantiles.  The shifted RMSE is uniform over the hard region -- "the
# scenes you actually need it to work in", not "the scenes you happened to get".
_U = np.linspace(0.0, 1.0, 201)
TRAIN_GRID = 1.0 - (1.0 - _U) ** (1.0 / (1.0 + SCENE_SKEW))
SHIFT_LO = 0.7
SHIFT_GRID = np.linspace(SHIFT_LO, 1.0, 101)

BETAS = [0.05, 0.1, 0.2, 0.4, 0.8, 2.0]
FLOORS = [0.0, 1e-4, 1e-3, 5e-3, 0.02, 0.1]
BETA_STAR = 0.4  # the weighting strength quoted in the strategy table
FLOOR_STAR = 5e-3  # the weight floor quoted in the strategy table


def expert(s):
    """The action a clean demonstration would execute at scene ``s``."""
    return np.sin(np.pi * s)


def draw_pool(rng):
    """One replica: a pool of demonstrations and the critic's view of it."""
    u = rng.uniform(0.0, 1.0, N_DEMOS)
    s = 1.0 - (1.0 - u) ** (1.0 / (1.0 + SCENE_SKEW))
    p_succ = np.clip(1.0 - s / SUCCESS_CEILING, P_SUCCESS_FLOOR, 1.0)
    success = rng.uniform(0.0, 1.0, N_DEMOS) < p_succ
    band = rng.uniform(0.0, 1.0, N_DEMOS)
    q = np.where(success, Q_SUCCESS_LO + (1.0 - Q_SUCCESS_LO) * band, Q_FAIL_HI * band)
    dev = (1.0 - q) * (FAIL_BIAS + FAIL_NOISE * rng.standard_normal(N_DEMOS))
    action = expert(s) + dev
    # The advantage is what the critic *thinks* it is: training-time noise, not a
    # property of the demonstration.
    adv = (2.0 * q - 1.0) + ADV_NOISE * rng.standard_normal(N_DEMOS)
    return {"s": s, "action": action, "q": q, "success": success, "adv": adv}


def weights(pool, kind, beta=BETA_STAR, floor=0.0):
    """The only thing that differs between the policies in this block."""
    if kind == "uniform":
        return np.ones(N_DEMOS)
    if kind == "filtered":
        # The critic's verdict, not the ground truth.  A perfect filter is a
        # different experiment and is reported separately.
        return (pool["adv"] > 0.0).astype(float)
    if kind == "oracle_filter":
        return pool["success"].astype(float)
    if kind in ("awr", "floor"):
        # Subtracting the max is a numerical convenience only: NW regression is
        # invariant to the overall scale of w, and exp(A/beta) overflows at the
        # small betas we sweep.
        z = pool["adv"] / beta
        w = np.exp(z - z.max())
        return np.maximum(w, floor) if floor > 0.0 else w
    raise ValueError(kind)  # pragma: no cover


def rmse(grid, pool, w):
    """Weighted kernel-regression prediction error against the expert action."""
    d = (grid[:, None] - pool["s"][None, :]) / BANDWIDTH
    k = np.exp(-0.5 * d * d) * w[None, :]
    pred = (k @ pool["action"]) / np.maximum(k.sum(axis=1), 1e-300)
    return float(np.sqrt(np.mean((pred - expert(grid)) ** 2)))


def ess_fraction(w):
    """Effective sample size over pool size: how many demonstrations the weights
    are really worth.  Equals the retention rate for a hard filter."""
    return float((w.sum() ** 2) / np.sum(w**2) / N_DEMOS)


def evaluate(pools, wf):
    train, shift, ess = [], [], []
    for pool in pools:
        w = wf(pool)
        train.append(rmse(TRAIN_GRID, pool, w))
        shift.append(rmse(SHIFT_GRID, pool, w))
        ess.append(ess_fraction(w))
    return float(np.mean(train)), float(np.mean(shift)), float(np.mean(ess))


def far_end_mechanics(pools):
    """Why the far end behaves the way it does.

    At the hardest evaluated scene, what fraction of the kernel's mass (weight
    included) sits on post-ceiling demonstrations -- i.e. on runs that all failed?
    And how many bandwidths away is the nearest *successful* demonstration?  The
    second number is the whole reason the first one is near one.
    """
    share, gap = [], []
    for pool in pools:
        w = weights(pool, "awr", beta=BETA_STAR)
        d = (SHIFT_GRID[-1] - pool["s"]) / BANDWIDTH
        k = np.exp(-0.5 * d * d) * w
        share.append(float(k[pool["s"] > SUCCESS_CEILING].sum() / k.sum()))
        ok = pool["s"][pool["success"].astype(bool)]
        gap.append(float((SHIFT_GRID[-1] - ok.max()) / BANDWIDTH))
    return float(np.mean(share)), float(np.mean(gap))


def constant_predictor(pools):
    """What a policy that learned nothing but the pool mean would score -- the
    line an over-filtered policy collapses towards."""
    out = []
    for pool in pools:
        mean_action = float(np.mean(pool["action"]))
        out.append(float(np.sqrt(np.mean((mean_action - expert(SHIFT_GRID)) ** 2))))
    return float(np.mean(out))


def main() -> int:
    pools = [draw_pool(mapkit.rng(SEED + 1 + k)) for k in range(N_REPLICAS)]

    rows, scores = [], {}
    for key, label, wf in [
        ("uniform", "全用", lambda p: weights(p, "uniform")),
        ("filtered", "只留成功", lambda p: weights(p, "filtered")),
        ("awr", "优势加权", lambda p: weights(p, "awr", beta=BETA_STAR)),
        ("awr_floor", "加权+保底", lambda p: weights(p, "floor", beta=BETA_STAR, floor=FLOOR_STAR)),
    ]:
        tr, sh, es = evaluate(pools, wf)
        scores[key] = (tr, sh, es)
        rows.append(
            {
                "strategy": key,
                "label": label,
                "effective_sample_pct": round(es * 100, 2),
                "train_rmse": round(tr, 5),
                "shift_rmse": round(sh, 5),
                "shift_over_train": round(sh / tr, 3),
            }
        )

    oracle_tr, oracle_sh, oracle_es = evaluate(pools, lambda p: weights(p, "oracle_filter"))

    beta_rows = []
    for beta in BETAS:
        tr, sh, es = evaluate(pools, lambda p, b=beta: weights(p, "awr", beta=b))
        beta_rows.append(
            {
                "beta": beta,
                "effective_sample_pct": round(es * 100, 2),
                "train_rmse": round(tr, 5),
                "shift_rmse": round(sh, 5),
            }
        )
    best_beta = min(beta_rows, key=lambda r: r["shift_rmse"])

    floor_rows = []
    for floor in FLOORS:
        tr, sh, es = evaluate(
            pools, lambda p, f=floor: weights(p, "floor", beta=BETA_STAR, floor=f)
        )
        floor_rows.append(
            {
                "floor": floor,
                "effective_sample_pct": round(es * 100, 2),
                "train_rmse": round(tr, 5),
                "shift_rmse": round(sh, 5),
            }
        )
    best_floor = min(floor_rows, key=lambda r: r["shift_rmse"])

    tr_u, sh_u, es_u = scores["uniform"]
    tr_f, sh_f, es_f = scores["filtered"]
    tr_a, sh_a, es_a = scores["awr"]
    tr_af, sh_af, es_af = scores["awr_floor"]
    flat = constant_predictor(pools)
    far_share, far_gap = far_end_mechanics(pools)

    metrics = {
        "n_demos": N_DEMOS,
        "n_replicas": N_REPLICAS,
        "scene_skew": SCENE_SKEW,
        "success_ceiling": SUCCESS_CEILING,
        "p_success_floor": P_SUCCESS_FLOOR,
        "q_success_lo": Q_SUCCESS_LO,
        "q_fail_hi": Q_FAIL_HI,
        "fail_bias": FAIL_BIAS,
        "fail_noise": FAIL_NOISE,
        "adv_noise": ADV_NOISE,
        "bandwidth": BANDWIDTH,
        "shift_lo": SHIFT_LO,
        "beta_star": BETA_STAR,
        "floor_star": FLOOR_STAR,
        # Mechanism, at the hardest evaluated scene.
        "far_end_failure_mass_pct": round(far_share * 100, 1),
        "nearest_success_gap_bandwidths": round(far_gap, 2),
        # The reference line: a policy that learned nothing at all.
        "constant_predictor_shift_rmse": round(flat, 5),
        # Headline: filtering looks like a win until the scenes move.
        "uniform_train_rmse": round(tr_u, 5),
        "uniform_shift_rmse": round(sh_u, 5),
        "filtered_train_rmse": round(tr_f, 5),
        "filtered_shift_rmse": round(sh_f, 5),
        "filtered_over_uniform_train": round(tr_u / tr_f, 3),
        "filtered_over_uniform_shift": round(sh_f / sh_u, 3),
        "awr_train_rmse": round(tr_a, 5),
        "awr_shift_rmse": round(sh_a, 5),
        "uniform_over_awr_train": round(tr_u / tr_a, 3),
        "uniform_over_awr_shift": round(sh_u / sh_a, 3),
        "awr_floor_train_rmse": round(tr_af, 5),
        "awr_floor_shift_rmse": round(sh_af, 5),
        # Effective sample size, in percent of the pool.
        "uniform_effective_sample_pct": round(es_u * 100, 2),
        "filtered_effective_sample_pct": round(es_f * 100, 2),
        "awr_effective_sample_pct": round(es_a * 100, 2),
        "awr_floor_effective_sample_pct": round(es_af * 100, 2),
        # How much filtering costs once the scenes move.
        "filtered_shift_over_train": round(sh_f / tr_f, 3),
        "awr_shift_over_train": round(sh_a / tr_a, 3),
        "uniform_shift_over_train": round(sh_u / tr_u, 3),
        # The knob sweep: how hard the weighting bites.
        "beta_sweep_min_rmse": best_beta["shift_rmse"],
        "best_beta": best_beta["beta"],
        "hardest_beta": BETAS[0],
        "hardest_beta_train_rmse": beta_rows[0]["train_rmse"],
        "hardest_beta_shift_rmse": beta_rows[0]["shift_rmse"],
        "hardest_beta_effective_sample_pct": beta_rows[0]["effective_sample_pct"],
        "hardest_over_uniform_shift": round(beta_rows[0]["shift_rmse"] / sh_u, 3),
        "hardest_over_uniform_train": round(beta_rows[0]["train_rmse"] / tr_u, 3),
        "hardest_over_best_beta_shift": round(
            beta_rows[0]["shift_rmse"] / best_beta["shift_rmse"], 3
        ),
        "softest_beta": BETAS[-1],
        "softest_beta_shift_rmse": beta_rows[-1]["shift_rmse"],
        "softest_beta_train_rmse": beta_rows[-1]["train_rmse"],
        # The knob sweep: how much bad data is bought back.  (Negative result.)
        "no_floor_shift_rmse": floor_rows[0]["shift_rmse"],
        "no_floor_train_rmse": floor_rows[0]["train_rmse"],
        "best_floor": best_floor["floor"],
        "best_floor_shift_rmse": best_floor["shift_rmse"],
        "best_floor_train_rmse": best_floor["train_rmse"],
        "floor_gain_shift": round(floor_rows[0]["shift_rmse"] / best_floor["shift_rmse"], 3),
        "full_floor_shift_rmse": floor_rows[-1]["shift_rmse"],
        "full_floor_effective_sample_pct": floor_rows[-1]["effective_sample_pct"],
        # The critic's imperfection, isolated: the same hard filter applied with
        # the true outcome instead of the estimated advantage.
        "oracle_filter_train_rmse": round(oracle_tr, 5),
        "oracle_filter_shift_rmse": round(oracle_sh, 5),
        "oracle_filter_effective_sample_pct": round(oracle_es * 100, 2),
        "critic_over_oracle_filter_shift": round(sh_f / oracle_sh, 3),
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"strategies": rows, "beta_sweep": beta_rows, "floor_sweep": floor_rows},
        seed=SEED,
        notes=(
            "A kernel smoother, not a trained policy, and an advantage "
            f"conditioning model, not a reproduction of RECAP or AWR. Means over "
            f"{N_REPLICAS} replicas of {N_DEMOS} demonstrations: the aggressive "
            "policies are high-variance, so one pool would not show the trade."
        ),
    )
    print(
        f"shift RMSE  uniform {sh_u:.4f} | filtered {sh_f:.4f} | awr {sh_a:.4f} "
        f"| awr+floor {sh_af:.4f}  (constant {flat:.4f}) -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
