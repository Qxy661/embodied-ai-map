#!/usr/bin/env python
"""Flow matching for action chunks, hand-written down to the backward pass.

Three claims, one experiment each:

(a) **The sign is not cosmetic.**  ``u = x1 - x0`` is the derivative of *this*
    path integrated in *this* direction.  Train the same net on the negated
    target and forward integration walks away from the data instead of onto it.

(b) **The step count has a sweet spot.**  One step is not a sampler, it is a
    regressor: it returns the conditional mean and lands in the empty middle of
    a bimodal action.  The modes only come back after a handful of steps.

(c) **L2 regression cannot do better than that.**  Train the same net to predict
    the action directly and every sample lands on the midpoint -- a point the
    data never visits.

The sampler itself is a 3-layer MLP, a straight-line path and a loop of
``x += dt * v``.  The backward pass is chain rule, ten lines of it, checked
against finite differences before a single step is taken.

The convention used here puts noise at tau=0.  The opposite orientation (noise at
t=1, target ``eps - A``, negative step) is just as valid -- but the sign of the
target and the direction of integration are a pair, and they have to travel
together.  That is claim (a), and it is the one thing in this file worth being
careful about.

    tau = 0 is noise, tau = 1 is data
    x_tau = tau * x1 + (1 - tau) * x0,  x1 ~ data,  x0 ~ N(0, I)
    u = d x_tau / d tau = x1 - x0
    inference: x <- x + dt * v, dt = 1 / steps, tau goes 0 -> 1
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0
DIM = 2
CENTERS = np.array([[1.5, 0.0], [-1.5, 0.0]])
CLUSTER_STD = 0.35
HIDDEN = 64
BATCH = 256
TRAIN_STEPS = 2000  # converged: the sampled distance stops moving well before this
LR = 1e-3
EVAL_N = 4096
SWEEP = (1, 2, 5, 10, 20, 50)
DEFAULT_STEPS = 10  # pi0's action-head step count
COVER_RADIUS = 1.0  # "on a mode" -- about 2.9 cluster sigmas


# --------------------------------------------------------------------------- #
# model: y = W3 tanh(W2 tanh(W1 [x, tau] + b1) + b2) + b3
# --------------------------------------------------------------------------- #
def init_params(rng: np.random.Generator) -> list[np.ndarray]:
    return [
        rng.standard_normal((DIM + 1, HIDDEN)) * (1.0 / np.sqrt(DIM + 1)),
        np.zeros(HIDDEN),
        rng.standard_normal((HIDDEN, HIDDEN)) * (1.0 / np.sqrt(HIDDEN)),
        np.zeros(HIDDEN),
        rng.standard_normal((HIDDEN, DIM)) * (1.0 / np.sqrt(HIDDEN)),
        np.zeros(DIM),
    ]


def forward(p: list[np.ndarray], X: np.ndarray):
    h1 = np.tanh(X @ p[0] + p[1])
    h2 = np.tanh(h1 @ p[2] + p[3])
    return h2 @ p[4] + p[5], (X, h1, h2)


def backward(p: list[np.ndarray], cache, dy: np.ndarray) -> list[np.ndarray]:
    """Chain rule, by hand.  ``dy`` is dL/dy for L = mean squared error."""
    X, h1, h2 = cache
    da2 = (dy @ p[4].T) * (1.0 - h2 * h2)
    da1 = (da2 @ p[2].T) * (1.0 - h1 * h1)
    return [
        X.T @ da1,
        da1.sum(axis=0),
        h1.T @ da2,
        da2.sum(axis=0),
        h2.T @ dy,
        dy.sum(axis=0),
    ]


def dloss_dy(y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """d/dy of mean((y - target)**2): averaged over outputs, not over rows."""
    return 2.0 * (y - target) / y.size


def grad_check() -> float:
    """Worst relative error between the analytic and the numerical gradient."""
    rng = mapkit.rng(SEED + 99)
    p = init_params(rng)
    X, u = make_batch("flow", rng, 16)

    def loss(params) -> float:
        y, _ = forward(params, X)
        return float(((y - u) ** 2).mean())

    y, cache = forward(p, X)
    analytic = backward(p, cache, dloss_dy(y, u))

    worst = 0.0
    for k, arr in enumerate(p):
        for flat in rng.choice(arr.size, size=min(8, arr.size), replace=False):
            idx = np.unravel_index(flat, arr.shape)
            orig = arr[idx]
            arr[idx] = orig + 1e-6
            hi = loss(p)
            arr[idx] = orig - 1e-6
            lo = loss(p)
            arr[idx] = orig
            num = (hi - lo) / 2e-6
            ana = analytic[k][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


# --------------------------------------------------------------------------- #
# data and training targets
# --------------------------------------------------------------------------- #
def sample_data(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two equal-weight Gaussians at -1.5 and +1.5: a bimodal action chunk."""
    return CENTERS[rng.integers(0, 2, size=n)] + CLUSTER_STD * rng.standard_normal((n, DIM))


def make_batch(mode: str, rng: np.random.Generator, n: int):
    """One training batch: network input X and the regression target u.

    mode="flow"     u = x1 - x0, the true path derivative
    mode="anti"     u = x0 - x1, the same field with the sign flipped
    mode="regress"  u = x1, tau pinned at 0 -- a plain L2 action regressor:
                    one shot from noise, no iteration and no interpolation
    """
    x1 = sample_data(rng, n)
    x0 = rng.standard_normal((n, DIM))
    if mode == "regress":
        return np.concatenate([x0, np.zeros((n, 1))], axis=1), x1
    tau = rng.random((n, 1))
    xt = tau * x1 + (1.0 - tau) * x0
    X = np.concatenate([xt, tau], axis=1)
    return X, (x1 - x0) if mode == "flow" else (x0 - x1)


def train(mode: str) -> tuple[list[np.ndarray], float]:
    """Adam on MSE.  Same seed and same data order for every mode, so the only
    thing that differs between two runs is the target they were fitted to."""
    p = init_params(mapkit.rng(SEED))
    rng = mapkit.rng(SEED + 1)
    m = [np.zeros_like(a) for a in p]
    v = [np.zeros_like(a) for a in p]
    loss = float("nan")
    for t in range(1, TRAIN_STEPS + 1):
        X, u = make_batch(mode, rng, BATCH)
        y, cache = forward(p, X)
        err = y - u
        g = backward(p, cache, dloss_dy(y, u))
        norm = np.sqrt(sum(float((gi**2).sum()) for gi in g))
        scale = min(1.0, 5.0 / (norm + 1e-12))  # clip; do not trust a hand-written grad
        for i in range(len(p)):
            gi = g[i] * scale
            m[i] = 0.9 * m[i] + 0.1 * gi
            v[i] = 0.999 * v[i] + 0.001 * gi**2
            p[i] = p[i] - LR * (m[i] / (1.0 - 0.9**t)) / (np.sqrt(v[i] / (1.0 - 0.999**t)) + 1e-8)
        loss = float((err**2).mean())
    return p, loss


# --------------------------------------------------------------------------- #
# sampling and measurement
# --------------------------------------------------------------------------- #
def field(p: list[np.ndarray], x: np.ndarray, tau: float) -> np.ndarray:
    v, _ = forward(p, np.concatenate([x, np.full((x.shape[0], 1), tau)], axis=1))
    return v


def sample(p: list[np.ndarray], rng: np.random.Generator, steps: int, n: int = EVAL_N):
    """Euler integration of dx/dtau = v(x, tau), tau going 0 -> 1."""
    x = rng.standard_normal((n, DIM))
    dt = 1.0 / steps
    for i in range(steps):
        x = x + dt * field(p, x, i * dt)
    return x


def regress_action(p: list[np.ndarray], rng: np.random.Generator, n: int = EVAL_N):
    """The regression policy's action: a = f(x0).  Nothing is added back."""
    return field(p, rng.standard_normal((n, DIM)), 0.0)


def stats(x: np.ndarray) -> dict:
    d = np.linalg.norm(x[:, None, :] - CENTERS[None, :, :], axis=2)
    near = d.argmin(axis=1)
    dmin = d.min(axis=1)
    return {
        "dist": float(dmin.mean()),
        "covered": float((dmin <= COVER_RADIUS).mean()),
        "gap": float((dmin > COVER_RADIUS).mean()),
        "share_plus": float((near == 0).mean()),
        "norm": float(np.linalg.norm(x, axis=1).mean()),
    }


def r6(x: float) -> float:
    return round(float(x), 6)


def main() -> int:
    gc_err = grad_check()
    assert gc_err < 1e-6, f"hand-written backward pass is wrong: rel err {gc_err:g}"

    flow, flow_loss = train("flow")
    anti, anti_loss = train("anti")
    regress, regress_loss = train("regress")

    rows = []
    at = {}
    for steps in SWEEP:
        s = stats(sample(flow, mapkit.rng(SEED + 100 + steps), steps))
        f = stats(sample(anti, mapkit.rng(SEED + 100 + steps), steps))
        at[steps] = (s, f)
        rows.append(
            {
                "steps": steps,
                "dist": r6(s["dist"]),
                "covered": r6(s["covered"]),
                "gap": r6(s["gap"]),
                "norm": r6(s["norm"]),
                "share_plus": r6(s["share_plus"]),
                "dist_flipped": r6(f["dist"]),
                "covered_flipped": r6(f["covered"]),
            }
        )

    good, bad = at[DEFAULT_STEPS]
    one = at[1][0]
    flipped = [r["dist_flipped"] for r in rows]

    # The regression baseline: one shot from noise, no iterative refinement.
    reg_x = regress_action(regress, mapkit.rng(SEED + 7))
    reg = stats(reg_x)

    # The data's own spread around its centres.  No sampler can beat this: it is
    # the width of the modes, not an error.  It is the yardstick the sweep is
    # measured against.
    x1 = sample_data(mapkit.rng(SEED + 8), EVAL_N)
    data = stats(x1)
    # What a sampler has to beat: the pure noise it starts from.
    noise = stats(mapkit.rng(SEED + 10).standard_normal((EVAL_N, DIM)))
    # Where the negated field walks: forward integration of -u follows the same
    # straight-line path, so it lands on the marginal at tau = -1, i.e. -x1 + 2*x0.
    rev = stats(-x1 + 2.0 * mapkit.rng(SEED + 9).standard_normal((EVAL_N, DIM)))

    metrics = {
        "dim": DIM,
        "center_offset": float(CENTERS[0, 0]),
        "cluster_std": CLUSTER_STD,
        "hidden": HIDDEN,
        "n_params": int(sum(a.size for a in flow)),
        "batch": BATCH,
        "train_steps": TRAIN_STEPS,
        "lr": LR,
        "eval_n": EVAL_N,
        "cover_radius": COVER_RADIUS,
        "data_norm": r6(data["norm"]),
        "data_dist": r6(data["dist"]),
        "data_covered": r6(data["covered"]),
        "noise_dist": r6(noise["dist"]),
        "noise_covered": r6(noise["covered"]),
        "grad_check_rel_err": round(float(gc_err), 10),
        "train_mse_flow": r6(flow_loss),
        "train_mse_anti": r6(anti_loss),
        "train_mse_regress": r6(regress_loss),
        # (b) step sweep, around the default step count
        "steps_default": DEFAULT_STEPS,
        "dist_at_1_step": r6(one["dist"]),
        "norm_at_1_step": r6(one["norm"]),
        "covered_at_1_step": r6(one["covered"]),
        "dist_at_default_steps": r6(good["dist"]),
        "norm_at_default_steps": r6(good["norm"]),
        "covered_at_default_steps": r6(good["covered"]),
        "dist_at_50_steps": r6(at[50][0]["dist"]),
        "covered_at_50_steps": r6(at[50][0]["covered"]),
        "dist_ratio_1_to_default": r6(one["dist"] / good["dist"]),
        "dist_ratio_default_to_50": r6(at[50][0]["dist"] / good["dist"]),
        # How much of the data's own spread / coverage the sampler has reached.
        "dist_vs_data_at_default": r6(good["dist"] / data["dist"]),
        "covered_vs_data_at_default": r6(good["covered"] / data["covered"]),
        # (a) the sign ablation
        "sign_correct_dist": r6(good["dist"]),
        "sign_correct_covered": r6(good["covered"]),
        "sign_flipped_dist": r6(bad["dist"]),
        "sign_flipped_covered": r6(bad["covered"]),
        "sign_flipped_dist_at_1_step": r6(flipped[0]),
        "sign_flipped_dist_at_50_steps": r6(flipped[-1]),
        "sign_flipped_growth_1_to_50": r6(flipped[-1] / flipped[0]),
        "sign_flipped_monotone": bool(np.all(np.diff(flipped) > 0)),
        "sign_penalty_ratio": r6(bad["dist"] / good["dist"]),
        "reversed_path_dist": r6(rev["dist"]),
        # (c) the L2 regressor
        "l2_dist": r6(reg["dist"]),
        "l2_gap_fraction": r6(reg["gap"]),
        "l2_covered": r6(reg["covered"]),
        "l2_norm": r6(reg["norm"]),
        "l2_norm_vs_data": r6(reg["norm"] / data["norm"]),
        "l2_output_std": r6(float(reg_x.std(axis=0).mean())),
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"step_sweep": rows},
        seed=SEED,
    )
    print(
        f"grad {gc_err:.1e} | 10 steps: dist {good['dist']:.3f} cover {good['covered']:.3f} "
        f"(data {data['dist']:.3f}/{data['covered']:.3f}) | flipped {bad['dist']:.3f} "
        f"| L2 gap {reg['gap']:.3f} dist {reg['dist']:.3f} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
