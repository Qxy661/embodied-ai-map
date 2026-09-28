#!/usr/bin/env python
"""Quantization, scored as a roofline question instead of a file-size question.

The belief this block exists to break: "INT8 makes the model smaller, therefore it
makes the model faster."  The first half is arithmetic.  The second half is a
property of the silicon, and the two are not the same claim.

For a single matmul of shape (M, K, N) we count

    flops = 2 * M * K * N
    bytes = w_bytes * K * N + a_bytes * (M * K + M * N)

and take the time to be the *larger* of the two limits the machine imposes:

    t = max(flops / peak(precision),  bytes / bandwidth)

That max() is the whole block.  Quantization moves both terms at once -- it shrinks
the byte count and it may or may not raise the peak -- and which one is binding
depends entirely on the shape.  Three schemes are compared against an FP16 baseline:

    w8a8              weights and activations to INT8.  Bytes x2, and (only if the
                      part actually has an INT8 tensor core) peak x2 as well.
    w4a16             weights to 4 bits, arithmetic left in FP16 -- the GPTQ/AWQ
                      trade.  Bytes shrink hard at small M, the peak never moves.
    w8a8_no_i8_core   INT8 on a part with no narrow datapath.  The peak does not
                      move and the narrow tensor has to be widened before the
                      multiply, so the widening pass pays back part of the win.

Nothing here is timed and nothing is measured.  It is a model of a machine, and the
only honest way to read it is as a map of *where* the win comes from -- which is
what you need before you spend a week quantizing something.  The companion block
⑤-02 asks the other half of the question: what the deployment stack costs you
regardless of how fast the arithmetic is.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0

# --- the machine (an edge-NPU class part, stated as an assumption) -----------
# These are the only two numbers the whole block rests on.  They are deliberately
# round: the conclusions have to survive being wrong about them by a factor of two,
# and one of the sweeps below shows they do.
PEAK_FP16 = 8.0e12  # FLOP/s, FP16 tensor core
BANDWIDTH = 68.0e9  # bytes/s, LPDDR5 class
RIDGE_FP16 = PEAK_FP16 / BANDWIDTH  # FLOP per byte: where the two limits cross

# --- the schemes ------------------------------------------------------------
# ``peak_ratio`` is a hardware property, not a quantization property.  W8A8 on a
# part with a narrow datapath is 2x FP16 (the usual tensor-core ratio); W4A16 on a
# part that has no INT4 datapath is 1x, because the multiply is still FP16.
#
# ``dequant_passes`` is the share of the quantized tensor's own volume that the
# widening pass has to move again.  It is 0 when the narrow datapath consumes the
# narrow tensor directly (the widen happens in registers and never touches DRAM),
# and it is not 0 when the runtime has to materialise the wide copy first.
SCHEMES = {
    "w8a8": {"w_bytes": 1.0, "a_bytes": 1.0, "peak_ratio": 2.0, "dequant_passes": 0.0},
    "w4a16": {"w_bytes": 0.5, "a_bytes": 2.0, "peak_ratio": 1.0, "dequant_passes": 0.0},
    "w8a8_no_i8_core": {"w_bytes": 1.0, "a_bytes": 1.0, "peak_ratio": 1.0, "dequant_passes": 2.0},
}

# The layer the batch sweep walks.  A single large square projection, so that the
# only thing changing across the sweep is how much work one fetched weight does.
SWEEP_K = 4096
SWEEP_N = 4096
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Concrete VLA shapes, written as (name, M, K, N).  M is the number of token
# positions in the batch: for a prefill that is the prompt length, for the action
# expert it is the number of action tokens in the chunk being decoded.
VLA_LAYERS = [
    ("vision patch embed", 1024, 768, 1024),
    ("vlm prefill ffn", 512, 4096, 14336),
    ("action expert ffn, chunk", 50, 1024, 1024),
    ("action expert ffn, one step", 1, 1024, 1024),
    ("action out proj", 50, 1024, 32),
]

# Peak-ratio sensitivity: how much of the win survives if the part's INT8 datapath
# is slower than the datasheet number, or absent.
PEAK_RATIOS = [1.0, 1.5, 2.0, 4.0]

# --- the error experiment ---------------------------------------------------
# A small action head, quantized for real.  Output dim is the action dimension: the
# error we care about is measured *after* the head, in action units, because that
# is where it stops being a numerics problem and starts being a control problem.
HEAD_DIMS = (64, 256, 256, 14)
Q_BITS = [8, 6, 5, 4, 3, 2]
N_SAMPLES = 256


def cost(
    m: int,
    k: int,
    n: int,
    scheme: str | None = None,
    peak_ratio: float | None = None,
    bandwidth: float = BANDWIDTH,
):
    """Roofline cost of one matmul.  ``scheme=None`` is the FP16 baseline."""
    s = SCHEMES[scheme] if scheme else None
    w_bytes = s["w_bytes"] if s else 2.0
    a_bytes = s["a_bytes"] if s else 2.0
    ratio = (s["peak_ratio"] if s else 1.0) if peak_ratio is None else peak_ratio

    flops = 2.0 * m * k * n
    nbytes = w_bytes * k * n + a_bytes * (m * k + m * n)
    # The widening pass does not create a tensor, it re-moves one; charging it as
    # extra traffic is the cheapest honest way to represent it in this grammar.
    effective = nbytes * (1.0 + (s["dequant_passes"] if s else 0.0))

    t_compute = flops / (PEAK_FP16 * ratio)
    t_memory = effective / bandwidth
    return {
        "flops": flops,
        "bytes": nbytes,
        "effective_bytes": effective,
        "arithmetic_intensity": flops / nbytes,
        "t_compute_s": t_compute,
        "t_memory_s": t_memory,
        "time_s": max(t_compute, t_memory),
        "bound": "compute" if t_compute >= t_memory else "memory",
    }


def speedup(
    m: int,
    k: int,
    n: int,
    scheme: str,
    peak_ratio: float | None = None,
    bandwidth: float = BANDWIDTH,
) -> float:
    base = cost(m, k, n, bandwidth=bandwidth)["time_s"]
    return base / cost(m, k, n, scheme, peak_ratio, bandwidth)["time_s"]


def crossover_batch(k: int, n: int, bandwidth: float = BANDWIDTH) -> float:
    """Batch at which the FP16 baseline stops being bandwidth-bound.

    Set 2*M*K*N / PEAK = (2*K*N + 2*M*(K+N)) / BW and solve for M.  Above this
    batch the layer has enough reuse that the multiply, not the fetch, is the wall.
    """
    num = (k * n) / bandwidth
    den = (k * n) / PEAK_FP16 - (k + n) / bandwidth
    return num / den if den > 0 else float("inf")


def batch_sweep() -> list[dict]:
    rows = []
    for m in BATCHES:
        base = cost(m, SWEEP_K, SWEEP_N)
        rows.append(
            {
                "M": m,
                "ai": round(base["arithmetic_intensity"], 4),
                "bound": base["bound"],
                "fp16_ms": round(base["time_s"] * 1e3, 6),
                "w8a8_ms": round(cost(m, SWEEP_K, SWEEP_N, "w8a8")["time_s"] * 1e3, 6),
                "w8a8_speedup": round(speedup(m, SWEEP_K, SWEEP_N, "w8a8"), 4),
                "w4a16_ms": round(cost(m, SWEEP_K, SWEEP_N, "w4a16")["time_s"] * 1e3, 6),
                "w4a16_speedup": round(speedup(m, SWEEP_K, SWEEP_N, "w4a16"), 4),
                "no_i8_core_speedup": round(speedup(m, SWEEP_K, SWEEP_N, "w8a8_no_i8_core"), 4),
            }
        )
    return rows


def peak_ratio_sweep() -> list[dict]:
    """Speedup vs the hardware's INT8/FP16 peak ratio, at both ends of the batch axis.

    The decode shape is the one that matters in a VLA: one token at a time, weights
    streamed from DRAM.  The prefill shape is the same matmul with a full prompt in
    the batch dimension.
    """
    rows = []
    for r in PEAK_RATIOS:
        rows.append(
            {
                "peak_ratio": r,
                "decode_M1_speedup": round(speedup(1, 1024, 1024, "w8a8", r), 4),
                "prefill_M512_speedup": round(speedup(512, 1024, 1024, "w8a8", r), 4),
                "decode_bound": cost(1, 1024, 1024, "w8a8", r)["bound"],
                "prefill_bound": cost(512, 1024, 1024, "w8a8", r)["bound"],
            }
        )
    return rows


def vla_table() -> list[dict]:
    rows = []
    for name, m, k, n in VLA_LAYERS:
        base = cost(m, k, n)
        rows.append(
            {
                "layer": name,
                "M": m,
                "K": k,
                "N": n,
                "ai": round(base["arithmetic_intensity"], 4),
                "bound": base["bound"],
                "fp16_ms": round(base["time_s"] * 1e3, 6),
                "w8a8_speedup": round(speedup(m, k, n, "w8a8"), 4),
                "w4a16_speedup": round(speedup(m, k, n, "w4a16"), 4),
            }
        )
    return rows


# --- error: what quantization does to the action, not to the weights ---------


def quantise(t: np.ndarray, bits: int, scale_axis: int | None = None) -> np.ndarray:
    """Symmetric uniform quantization with an exact (round-to-nearest) encoder.

    ``scale_axis`` says how finely the scale is allowed to adapt, and it is the one
    dial that decides whether low bit widths are usable at all:

      None  one scale for the whole tensor (per-tensor)
      0     one scale per output channel (per-channel -- the weight case)
      -1    one scale per row (per-token -- the activation case)

    The activation axis is the batch axis here: each sample's hidden vector carries
    its own scale, which is the same thing a per-token dynamic quantizer does.
    """
    qmax = float(2 ** (bits - 1) - 1)
    if scale_axis is None:
        scale = np.abs(t).max() / qmax
    else:
        scale = np.abs(t).max(axis=scale_axis, keepdims=True) / qmax
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.clip(np.round(t / scale), -qmax, qmax) * scale


def head(seed: int = SEED):
    """A deterministic action head: 64 -> 256 -> 256 -> 14, He-initialised."""
    r = mapkit.rng(seed)
    ws = []
    for fan_in, fan_out in zip(HEAD_DIMS[:-1], HEAD_DIMS[1:], strict=True):
        ws.append(r.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
    return ws


def run_head(ws, x, bits=None, layers=(), q_weights=True, q_acts=True, w_axis=0):
    """Forward pass over a whole batch of inputs at once.

    ``layers`` selects which layers get quantized at all, which is what the
    sensitivity sweep varies.  ``q_weights`` / ``q_acts`` split the two halves of
    W8A8, which is what the mixed-precision comparison varies: a W4A16 kernel
    shrinks the weights and leaves the activations -- and the arithmetic -- alone.
    ``w_axis=None`` degrades the weight scale from per-channel to per-tensor.
    """
    h = x
    for i, w in enumerate(ws):
        if bits is not None and i in layers:
            if q_weights:
                w = quantise(w, bits, w_axis)
            if q_acts:
                h = quantise(h, bits, -1)
        h = h @ w
        if i < len(ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def error_experiment():
    ws = head()
    r = mapkit.rng(SEED + 1)
    xs = r.normal(0.0, 1.0, (N_SAMPLES, HEAD_DIMS[0]))

    def simulate(**kw):
        return run_head(ws, xs, layers=(0, 1, 2), **kw)

    ref = run_head(ws, xs)
    ref_std = float(ref.std())

    def mae(a):
        """Mean action error as a percentage of the reference action spread.

        Measured after the head, not on the weights.  A weight error is a numerics
        number; an action error is a controls number, and only one of them is what
        the robot feels.
        """
        return round(float(np.abs(a - ref).mean()) / ref_std * 100.0, 6)

    bits_rows = []
    for b in Q_BITS:
        w_pc = simulate(bits=b, q_acts=False, w_axis=0)
        w_pt = simulate(bits=b, q_acts=False, w_axis=None)
        full = simulate(bits=b, q_acts=True, w_axis=0)
        bits_rows.append(
            {
                "bits": b,
                # weights only -- the W8A16 / W4A16 shape, arithmetic untouched
                "w_per_channel_pct": mae(w_pc),
                "w_per_tensor_pct": mae(w_pt),
                # the full W8A8 shape: per-channel weights, per-tensor activations
                "w_plus_a_pct": mae(full),
                # how much of the total error the activations add on top of the weights
                "activation_share": round(mae(full) / mae(w_pc), 3) if mae(w_pc) > 0 else None,
            }
        )

    # Sensitivity: quantize exactly one layer at a time.  The question is where the
    # error is cheapest to spend, and the answer is not symmetric across the stack.
    sens = []
    for i in range(len(ws)):
        sens.append(
            {
                "layer": i,
                "fan_in": HEAD_DIMS[i],
                "fan_out": HEAD_DIMS[i + 1],
                "int8_pct": mae(run_head(ws, xs, bits=8, layers=(i,))),
                "int4_pct": mae(run_head(ws, xs, bits=4, layers=(i,))),
            }
        )
    return ref_std, bits_rows, sens


def scheme_table() -> list[dict]:
    """The model's inputs, emitted so the prose is allowed to name them."""
    return [{"scheme": k, **v} for k, v in SCHEMES.items()]


def bandwidth_sweep() -> list[dict]:
    """The one place the machine's numbers are allowed to move.

    Both hardware constants are assumptions, so the honest thing is to show what
    follows them.  The ridge point is peak over bandwidth; the batch at which a
    layer stops being bandwidth-bound follows the ridge point.
    """
    rows = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        bw = BANDWIDTH * scale
        rows.append(
            {
                "bandwidth_gbs": round(bw / 1e9, 3),
                "ridge_flops_per_byte": round(PEAK_FP16 / bw, 4),
                "m_crossover": round(crossover_batch(SWEEP_K, SWEEP_N, bw), 2),
                # the decode shape is memory-bound at every one of these bandwidths,
                # so its speedup does not move
                "decode_speedup": round(speedup(1, 1024, 1024, "w8a8", bandwidth=bw), 4),
            }
        )
    return rows


def main() -> int:
    sweep = batch_sweep()
    ratios = peak_ratio_sweep()
    vla = vla_table()
    bws = bandwidth_sweep()
    ref_std, err_bits, sens = error_experiment()

    by_m = {r["M"]: r for r in sweep}
    cross = crossover_batch(SWEEP_K, SWEEP_N)

    w8a8 = [r["w8a8_speedup"] for r in sweep]

    decode = next(r for r in vla if r["M"] == 1)
    prefill = max(vla, key=lambda r: r["K"] * r["N"])
    memory_bound = [r for r in vla if r["bound"] == "memory"]

    weights_bytes = sum(k * n for _, _, k, n in VLA_LAYERS)
    err8 = next(r for r in err_bits if r["bits"] == 8)
    err4 = next(r for r in err_bits if r["bits"] == 4)
    worst = max(sens, key=lambda s: s["int4_pct"])
    best = min(sens, key=lambda s: s["int4_pct"])

    metrics = {
        "peak_fp16_tflops": round(PEAK_FP16 / 1e12, 3),
        "bandwidth_gbs": round(BANDWIDTH / 1e9, 3),
        # The judge.  A layer above this FLOP:byte ratio is compute-bound and only
        # a faster peak helps it; below it, only a smaller byte count helps.
        "ridge_flops_per_byte": round(RIDGE_FP16, 4),
        # The same two hardware constants, wrong by 2x and 4x in either direction.
        "m_crossover_at_half_bandwidth": bws[0]["m_crossover"],
        "m_crossover_at_4x_bandwidth": bws[-1]["m_crossover"],
        "decode_speedup_across_bandwidth_octave": round(
            max(r["decode_speedup"] for r in bws) - min(r["decode_speedup"] for r in bws), 4
        ),
        "sweep_k": SWEEP_K,
        "sweep_n": SWEEP_N,
        # How much smaller the weights get.  This half is not in doubt.
        "weight_bytes_fp32_mb": round(weights_bytes * 4 / 1e6, 3),
        "weight_bytes_fp16_mb": round(weights_bytes * 2 / 1e6, 3),
        "weight_bytes_int8_mb": round(weights_bytes * 1 / 1e6, 3),
        "size_reduction_vs_fp32": 4.0,
        "size_reduction_vs_fp16": 2.0,
        # Batch sweep.  The flat column is the finding.
        "m_crossover_fp16": round(cross, 2),
        "ai_at_M1": by_m[1]["ai"],
        "ai_at_M4096": by_m[4096]["ai"],
        "w8a8_speedup_min": min(w8a8),
        "w8a8_speedup_max": max(w8a8),
        "w8a8_speedup_spread": round(max(w8a8) - min(w8a8), 4),
        "w8a8_flat": max(w8a8) == min(w8a8),
        "w4a16_speedup_at_M1": by_m[1]["w4a16_speedup"],
        "w4a16_speedup_at_M4096": by_m[4096]["w4a16_speedup"],
        "w4a16_gain_decay": round(by_m[1]["w4a16_speedup"] / by_m[4096]["w4a16_speedup"], 3),
        "no_i8_core_speedup_at_M1": by_m[1]["no_i8_core_speedup"],
        "no_i8_core_speedup_at_M4096": by_m[4096]["no_i8_core_speedup"],
        # Peak-ratio sweep: at M=1 the speedup is pinned by bandwidth and does not
        # care what the tensor core can do.
        "decode_speedup_peak_ratio_1": ratios[0]["decode_M1_speedup"],
        "decode_speedup_peak_ratio_4": ratios[-1]["decode_M1_speedup"],
        "prefill_speedup_peak_ratio_1": ratios[0]["prefill_M512_speedup"],
        "prefill_speedup_peak_ratio_4": ratios[-1]["prefill_M512_speedup"],
        # VLA shapes.
        "vla_layers": len(vla),
        "vla_memory_bound_layers": len(memory_bound),
        "vla_min_ai": min(r["ai"] for r in vla),
        "vla_max_ai": max(r["ai"] for r in vla),
        # The layer where the whole bandwidth story stops applying: a few
        # microseconds is launch overhead territory, not DRAM territory.
        "smallest_layer_us": round(min(r["fp16_ms"] for r in vla) * 1e3, 4),
        "largest_layer_ms": round(max(r["fp16_ms"] for r in vla), 4),
        "decode_layer_ai": decode["ai"],
        "decode_layer_w8a8_speedup": decode["w8a8_speedup"],
        "prefill_layer_ai": prefill["ai"],
        "prefill_layer_w8a8_speedup": prefill["w8a8_speedup"],
        # Error, in action units.  Everything below is measured after the head.
        "action_ref_std": round(ref_std, 6),
        "action_mae_int8_pct": err8["w_plus_a_pct"],
        "action_mae_int8_weights_only_pct": err8["w_per_channel_pct"],
        "action_mae_int4_pct": err4["w_plus_a_pct"],
        "action_mae_int4_weights_only_pct": err4["w_per_channel_pct"],
        "action_mae_int4_per_tensor_pct": err4["w_per_tensor_pct"],
        "activation_share_at_8bit": err8["activation_share"],
        "error_growth_8_to_4_bit": round(err4["w_plus_a_pct"] / err8["w_plus_a_pct"], 1),
        "per_channel_gain_at_4bit": round(err4["w_per_tensor_pct"] / err4["w_per_channel_pct"], 3),
        "worst_layer_at_4bit": worst["layer"],
        "worst_layer_int4_pct": worst["int4_pct"],
        "best_layer_at_4bit": best["layer"],
        "best_layer_int4_pct": best["int4_pct"],
        "layer_spread_at_4bit": round(worst["int4_pct"] / best["int4_pct"], 3),
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {
            "schemes": scheme_table(),
            "batch_sweep": sweep,
            "peak_ratio_sweep": ratios,
            "vla_layers": vla,
            "bandwidth_sweep": bws,
            "error_bits": err_bits,
            "layer_sensitivity": sens,
        },
        seed=SEED,
        notes=(
            "Pure arithmetic roofline plus a seeded quantization of a synthetic action "
            "head. No timing, no device: the machine is a declared assumption "
            f"({PEAK_FP16 / 1e12:.0f} TFLOP/s FP16, {BANDWIDTH / 1e9:.0f} GB/s)."
        ),
    )
    print(
        f"w8a8 flat x{min(w8a8):.2f} over M=1..4096 | w4a16 x{by_m[1]['w4a16_speedup']:.2f} -> "
        f"x{by_m[4096]['w4a16_speedup']:.2f} | ridge {RIDGE_FP16:.1f} | -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
