#!/usr/bin/env python
"""Autoregressive action tokens: what discrete bins and a serial decode cost.

③-01 established the frequency mismatch.  ③-02 filled it with a continuous action
head that reaches the answer in a handful of parallel integration steps.  This block
walks the third route, the one RT-2 / OpenVLA take: cut every action dimension into
N uniform bins and then emit the chunk **one token at a time**, the way a language
model emits a sentence.

The route buys a lot.  Next-token prediction, KV cache, teacher forcing, one shared
vocabulary for pixels, words and torques, and a decade of LLM engineering that
arrives for free.  It charges in two currencies, and this file prices both.

  quantisation   N bins cannot represent a continuous target.  The floor of that
                 error is set by the encoder's **range**, not by the action's own
                 scale, so a policy that only ever moves a little still pays the
                 full bin width.  Measured against the uniform-quantiser law, which
                 for a round-to-nearest mid-tread quantiser is step/sqrt(12).

  accumulation   a decode step conditioned on its own prefix inherits the prefix's
                 error.  The whole family of behaviours -- bounded, damped, drifting
                 -- is one recursion with one knob:

                     e_k = eta_k + rho * e_{k-1}

                 rho = 0 is "every token re-anchored on the observation", rho = 1 is
                 "the token stream is the only carrier of the plan", which is exactly
                 a delta tokenisation.  Everything real sits between.

Two trace families are run, because the second one is where the textbook answer turns
out to be wrong.  ``periodic`` is a single sinusoid near 1 Hz -- the motion ③-01 picks,
and a fair stand-in for a robot repeating a taught trajectory.  ``bandlimited`` sums
five incommensurate sinusoids, so the trace never repeats and neither do its increments.

Neither currency is measured on a trained network.  The decoder is that explicit error
recursion, not a model fitted to anything; eta is the real quantisation error of a real
action trace, and rho is a stated assumption.  The README says so again in 边界, because
a number that looks like a benchmark and is not one is worse than no number.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0
RATE_HZ = 50.0  # one action token per control step, the AR-native regime
FREQ_LO = 0.8  # Hz, the 1 Hz motion of ③-01 with a margin either side
FREQ_HI = 1.2
FREQ_NOMINAL = 1.0
AMP = 0.4  # amplitude, in units of the encoder's full range
EPISODES = 512
K_MAX = 128  # longest chunk we trace the error over, in tokens
CHUNK_K = 50  # the chunk length ③-01's feasibility bound forces you to use
BINS = [16, 32, 64, 128, 256, 512, 1024]
RHOS = [0.0, 0.9, 0.99, 1.0]
FLOORS = [0.002, 0.005, 0.01, 0.02]  # execution-noise floors, of the full range
FLOOR_HEADLINE = 0.01
ACTION_DIM = 7  # a typical arm: 6 joints + gripper
FM_STEPS = 10  # pi0's action-head step count, the ③-02 baseline
FIT_LO = 16  # log-log fit window for the growth exponent, in tokens
FIT_HI = 128
N_TABLE = 256  # the bin count the accumulation table is reported at
HARMONICS = (1.0, 2.7, 4.9, 7.3, 11.1)  # incommensurate ratios, for the aperiodic trace
FAMILIES = ("periodic", "bandlimited")
SQRT12 = float(np.sqrt(12.0))


def action_pool(family: str) -> np.ndarray:
    """(EPISODES, K_MAX+1) one action dimension, sampled at RATE_HZ.

    Both families are normalised to the same peak amplitude AMP, so the *range* an
    absolute encoder needs is the same for both.  What differs is the spectrum: the
    periodic trace's increments repeat every period, the bandlimited one's do not.
    That is the only knob this experiment varies, and it is the one that decides
    whether quantisation error accumulates like a random walk or cancels.
    """
    g = mapkit.rng(SEED)
    f = g.uniform(FREQ_LO, FREQ_HI, EPISODES)
    phase = g.uniform(0.0, 2.0 * np.pi, EPISODES)
    t = np.arange(K_MAX + 1) / RATE_HZ
    if family == "periodic":
        sig = np.sin(2.0 * np.pi * f[:, None] * t[None, :] + phase[:, None])
    else:
        sig = np.zeros((EPISODES, t.size))
        for ratio in HARMONICS:
            c = 1.0 / ratio**1.2  # decay so the trace stays smooth, like a real one
            ph = g.uniform(0.0, 2.0 * np.pi, EPISODES)
            sig += c * np.sin(2.0 * np.pi * (f * ratio)[:, None] * t[None, :] + ph[:, None])
        sig /= np.abs(sig).max(axis=1, keepdims=True)  # same peak as the periodic trace
    return AMP * sig


def quantise(x: np.ndarray, half_range: float, n: int) -> tuple[np.ndarray, float]:
    """Round-to-nearest uniform mid-tread quantiser over [-half_range, +half_range]."""
    step = 2.0 * half_range / n
    idx = np.clip(np.round((x + half_range) / step), 0.0, n - 1.0)
    return idx * step - half_range, step


def recurse(eta: np.ndarray, rho: float, e0: np.ndarray | None = None) -> np.ndarray:
    """e_k = eta_k + rho * e_{k-1}, over a (EPISODES, K) block of error increments."""
    out = np.empty_like(eta)
    prev = np.zeros(eta.shape[0]) if e0 is None else e0
    for k in range(eta.shape[1]):
        prev = eta[:, k] + rho * prev
        out[:, k] = prev
    return out


def rms_by_step(err: np.ndarray) -> np.ndarray:
    return np.sqrt((err**2).mean(axis=0))


def exponent(curve: np.ndarray) -> float:
    """Least-squares slope of log(RMS) vs log(k).  0.5 is the random-walk law."""
    k = np.arange(1, curve.size + 1)
    sel = (k >= FIT_LO) & (k <= FIT_HI)
    return float(np.polyfit(np.log(k[sel].astype(float)), np.log(curve[sel]), 1)[0])


def build(family: str):
    """Ranges, quantisation table and error curves for one trace family."""
    a = action_pool(family)
    pos = a[:, 1:]  # steps 1..K_MAX: what an absolute token encodes
    delta = np.diff(a, axis=1)  # steps 1..K_MAX: what a delta token encodes
    anchor = a[:, 0]  # a delta scheme still needs one absolute token to start

    # Encoder ranges, set the way RT-2 / OpenVLA set them: per-dimension min/max over
    # the dataset, frozen at training time.  The *extremes* of the data decide the bin
    # width, never its typical scale -- which is why a gently-moving policy still pays
    # a coarse bin.
    abs_range = float(np.abs(a).max())
    delta_range = float(np.abs(delta).max())

    rows, curves = [], {}
    for n in BINS:
        qa, abs_step = quantise(pos, abs_range, n)
        qd, del_step = quantise(delta, delta_range, n)
        eta_abs, eta_del = qa - pos, qd - delta
        q0, _ = quantise(anchor, abs_range, n)
        e0 = q0 - anchor

        sd_abs, sd_del = float(eta_abs.std()), float(eta_del.std())
        law_abs, law_del = abs_step / SQRT12, del_step / SQRT12
        rows.append(
            {
                "family": family,
                "N": n,
                "abs_step": round(abs_step, 8),
                "delta_step": round(del_step, 8),
                "abs_rms_measured": round(sd_abs, 8),
                "abs_rms_uniform_law": round(law_abs, 8),
                "abs_measured_over_law": round(sd_abs / law_abs, 4),
                "delta_rms_measured": round(sd_del, 8),
                "delta_rms_uniform_law": round(law_del, 8),
                "delta_measured_over_law": round(sd_del / law_del, 4),
            }
        )
        for rho in RHOS:
            curves[f"abs_rho{rho:g}_N{n}"] = rms_by_step(recurse(eta_abs, rho))
        curves[f"delta_rho1_N{n}"] = rms_by_step(recurse(eta_del, 1.0, e0))
        # The same integration with the anchor removed, so the error of the first
        # token (quantised on the position scale) can be told apart from the error
        # the increments accumulate (quantised on the delta scale).
        curves[f"delta_incr_N{n}"] = rms_by_step(recurse(eta_del, 1.0))

    return a, abs_range, delta_range, rows, curves


def main() -> int:
    built = {fam: build(fam) for fam in FAMILIES}
    quant_rows = [r for fam in FAMILIES for r in built[fam][3]]

    # --------------------------------------------------------------------- #
    # Accumulation: one recursion, two tokenisations at its two ends.
    #   absolute tokens, rho < 1  ->  the current observation re-anchors each step
    #   delta tokens,    rho = 1  ->  exact integration of the emitted increments
    # The textbook law at rho = 1 is k**0.5: independent errors, a random walk.
    # Whether that law holds is an empirical question, and the answer differs by
    # family -- that is the finding.
    # --------------------------------------------------------------------- #
    acc_rows = []
    for fam in FAMILIES:
        curves = built[fam][4]
        for rho in RHOS:
            c = curves[f"abs_rho{rho:g}_N{N_TABLE}"]
            acc_rows.append(
                {
                    "family": fam,
                    "scheme": "absolute",
                    "rho": rho,
                    "N": N_TABLE,
                    "rms_at_k1": round(float(c[0]), 8),
                    "rms_at_k16": round(float(c[FIT_LO - 1]), 8),
                    "rms_at_k50": round(float(c[CHUNK_K - 1]), 8),
                    "rms_at_k128": round(float(c[-1]), 8),
                    "growth_k1_to_k128": round(float(c[-1] / c[0]), 4),
                    "fitted_exponent": round(exponent(c), 4),
                }
            )
        for scheme, e0_zero in (("delta", False), ("delta_increments_only", True)):
            c = (
                curves[f"delta_rho1_N{N_TABLE}"]
                if not e0_zero
                else curves[f"delta_incr_N{N_TABLE}"]
            )
            acc_rows.append(
                {
                    "family": fam,
                    "scheme": scheme,
                    "rho": 1.0,
                    "N": N_TABLE,
                    "rms_at_k1": round(float(c[0]), 8),
                    "rms_at_k16": round(float(c[FIT_LO - 1]), 8),
                    "rms_at_k50": round(float(c[CHUNK_K - 1]), 8),
                    "rms_at_k128": round(float(c[-1]), 8),
                    "growth_k1_to_k128": round(float(c[-1] / c[0]), 4),
                    "fitted_exponent": round(exponent(c), 4),
                }
            )

    # --------------------------------------------------------------------- #
    # How many bins before quantisation stops being the bottleneck?  The judge is
    # the execution noise floor: everything the policy does not model -- contact,
    # payload, calibration, servo compliance.  Any error below that floor is free.
    # --------------------------------------------------------------------- #
    curves_p = built["periodic"][4]
    floor_rows = []
    for floor in FLOORS:
        row: dict[str, object] = {"floor": floor}
        for scheme in ("abs_rho0", "abs_rho0.99", "abs_rho1", "delta_rho1"):
            hit = None
            for n in BINS:
                if float(curves_p[f"{scheme}_N{n}"][CHUNK_K - 1]) < floor:
                    hit = n
                    break
            row[f"{scheme}_min_bins"] = hit
            row[f"{scheme}_rms_at_chunk_end_256"] = round(
                float(curves_p[f"{scheme}_N{N_TABLE}"][CHUNK_K - 1]), 8
            )
        # Closed form for the no-accumulation case: step/sqrt(12) < floor.  The
        # sweep can only report a power of two, so this is what the grid rounds up to.
        row["abs_rho0_min_bins_closed_form"] = int(
            np.ceil(2.0 * built["periodic"][1] / (floor * SQRT12))
        )
        floor_rows.append(row)

    # --------------------------------------------------------------------- #
    # The other currency: chunks are emitted serially, one token at a time.  No
    # timing is measured -- the ratio below is pure structure.  A flow-matching head
    # pays FM_STEPS forwards for the whole chunk; an AR head pays one forward per
    # token, and its per-control-step cost does not fall as the chunk grows.
    # --------------------------------------------------------------------- #
    latency_rows = []
    for k in (1, 4, 16, 50):
        tokens = k * ACTION_DIM  # 7 dims, one token each per control step
        latency_rows.append(
            {
                "chunk_steps": k,
                "ar_tokens_per_chunk": tokens,
                "ar_forwards_per_chunk": tokens,
                "fm_forwards_per_chunk": FM_STEPS,
                "serial_ratio_vs_fm": round(tokens / FM_STEPS, 4),
                "ar_forwards_per_control_step": ACTION_DIM,
                "fm_forwards_per_control_step": round(FM_STEPS / k, 6),
            }
        )

    def acc(fam: str, scheme: str, rho: float) -> dict:
        return next(
            r for r in acc_rows if r["family"] == fam and r["scheme"] == scheme and r["rho"] == rho
        )

    p_abs0 = acc("periodic", "absolute", 0.0)
    p_abs99 = acc("periodic", "absolute", 0.99)
    p_abs1 = acc("periodic", "absolute", 1.0)
    p_del = acc("periodic", "delta", 1.0)
    p_incr = acc("periodic", "delta_increments_only", 1.0)
    b_del = acc("bandlimited", "delta", 1.0)
    b_incr = acc("bandlimited", "delta_increments_only", 1.0)
    b_abs1 = acc("bandlimited", "absolute", 1.0)
    by_n = {r["N"]: r for r in quant_rows if r["family"] == "periodic"}
    floor_h = next(r for r in floor_rows if r["floor"] == FLOOR_HEADLINE)

    # The delta scheme still spends one token on an absolute position to start from,
    # and that token is quantised on the position scale -- the coarse one.  Measure
    # it separately: it is re-paid on every chunk, so it does not amortise.
    a_p = built["periodic"][0]
    q0, _ = quantise(a_p[:, 0], built["periodic"][1], N_TABLE)
    anchor_rms = float(np.sqrt(((q0 - a_p[:, 0]) ** 2).mean()))

    metrics = {
        "rate_hz": RATE_HZ,
        "freq_lo_hz": FREQ_LO,
        "freq_hi_hz": FREQ_HI,
        "freq_nominal_hz": FREQ_NOMINAL,
        "amp_of_range": AMP,
        "episodes": EPISODES,
        "k_max": K_MAX,
        "chunk_k": CHUNK_K,
        "action_dim": ACTION_DIM,
        "fm_steps": FM_STEPS,
        "harmonics": len(HARMONICS),
        # Encoder ranges and their ratio: the one number that decides whether a delta
        # tokenisation is worth anything at all.
        "abs_range": round(built["periodic"][1], 8),
        "delta_range_periodic": round(built["periodic"][2], 8),
        "delta_range_bandlimited": round(built["bandlimited"][2], 8),
        "range_ratio_periodic": round(built["periodic"][1] / built["periodic"][2], 4),
        # Quantisation, at the bin count the field actually ships.
        "abs_rms_at_N256": by_n[256]["abs_rms_measured"],
        "delta_rms_per_step_at_N256": by_n[256]["delta_rms_measured"],
        "abs_measured_over_law_at_N256": by_n[256]["abs_measured_over_law"],
        "abs_measured_over_law_at_N16": by_n[16]["abs_measured_over_law"],
        "delta_measured_over_law_at_N256": by_n[256]["delta_measured_over_law"],
        "delta_measured_over_law_at_N16": by_n[16]["delta_measured_over_law"],
        # Accumulation exponents.  0.5 is the random-walk law; none of them is it.
        "exponent_abs_rho0_periodic": p_abs0["fitted_exponent"],
        "exponent_abs_rho099_periodic": p_abs99["fitted_exponent"],
        "exponent_abs_rho1_periodic": p_abs1["fitted_exponent"],
        "exponent_delta_periodic": p_del["fitted_exponent"],
        "exponent_delta_bandlimited": b_del["fitted_exponent"],
        "exponent_delta_increments_periodic": p_incr["fitted_exponent"],
        "exponent_delta_increments_bandlimited": b_incr["fitted_exponent"],
        "exponent_abs_rho1_bandlimited": b_abs1["fitted_exponent"],
        "random_walk_exponent": 0.5,
        # What a random walk predicts for the ratio between the first and the last
        # token of the traced chunk, against which the measured growth is read.
        "random_walk_growth_k1_to_k128": round(float(np.sqrt(K_MAX)), 4),
        "growth_k1_to_k128_delta_increments_periodic": p_incr["growth_k1_to_k128"],
        "growth_k1_to_k128_delta_increments_bandlimited": b_incr["growth_k1_to_k128"],
        "growth_k1_to_k128_delta_periodic": p_del["growth_k1_to_k128"],
        "growth_k1_to_k128_delta_bandlimited": b_del["growth_k1_to_k128"],
        "growth_k1_to_k128_abs_rho1_periodic": p_abs1["growth_k1_to_k128"],
        "delta_over_abs_at_chunk_end": round(p_del["rms_at_k50"] / p_abs0["rms_at_k50"], 4),
        "delta_over_abs_at_k1": round(p_del["rms_at_k1"] / p_abs0["rms_at_k1"], 4),
        # The anchor token the delta scheme has to spend, and the decomposition of
        # its error into "anchor" and "increments".
        "anchor_rms_at_N256": round(anchor_rms, 8),
        "delta_increments_rms_at_k50": p_incr["rms_at_k50"],
        "delta_full_rms_at_k50": p_del["rms_at_k50"],
        "anchor_to_delta_per_step": round(anchor_rms / by_n[256]["delta_rms_measured"], 4),
        "anchor_share_of_delta_error": round(anchor_rms / p_del["rms_at_k50"], 4),
        # The judgement call, at 1% of range.
        "floor_headline": FLOOR_HEADLINE,
        "min_bins_abs_rho0": floor_h["abs_rho0_min_bins"],
        "min_bins_abs_rho099": floor_h["abs_rho0.99_min_bins"],
        "min_bins_abs_rho1": floor_h["abs_rho1_min_bins"],
        "min_bins_delta_rho1": floor_h["delta_rho1_min_bins"],
        "min_bins_abs_rho0_closed_form": floor_h["abs_rho0_min_bins_closed_form"],
        # Latency, structural only.
        "ar_forwards_per_chunk_k50": latency_rows[-1]["ar_forwards_per_chunk"],
        "serial_ratio_vs_fm_k50": latency_rows[-1]["serial_ratio_vs_fm"],
        "serial_ratio_vs_fm_k16": latency_rows[2]["serial_ratio_vs_fm"],
    }

    tables = {
        "quantisation": quant_rows,
        "accumulation": acc_rows,
        "bins_vs_noise_floor": floor_rows,
        "serial_latency": latency_rows,
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        tables,
        seed=SEED,
        notes=(
            "The decoder is an explicit error recursion e_k = eta_k + rho*e_{k-1}, not a "
            "trained network.  eta is the measured quantisation error of a real action "
            "trace; rho is a stated assumption about how much the decoder leans on its "
            "own prefix.  Read the numbers as the mechanism priced out, not as a "
            "benchmark of any published checkpoint."
        ),
    )
    print(
        f"exponent delta: periodic {p_del['fitted_exponent']:.3f} vs bandlimited "
        f"{b_del['fitted_exponent']:.3f} (textbook {metrics['random_walk_exponent']}) | "
        f"min bins @1%: abs {floor_h['abs_rho0_min_bins']}, "
        f"self-cond {floor_h['abs_rho0.99_min_bins']}, "
        f"delta {floor_h['delta_rho1_min_bins']} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
