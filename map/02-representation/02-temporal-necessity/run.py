#!/usr/bin/env python
"""Does a single BEV frame carry velocity?  Measured, not argued.

Eight actors on straight constant-velocity tracks inside a 64x64 occupancy grid at
0.5 m per cell.  Frames are rasterised the way a BEV head emits them -- perturb the
true position, paint an occupancy blob, threshold -- and *every* estimator reads
nothing but that grid.  No network, no tracker, no motion model beyond ``p + v*t``.

  single frame    no time axis, so the only honest report is zero.  Its error is the
                  scene's whole mean speed, and on a frame with no detection it has
                  nothing to say at all.
  K-frame slope   least-squares slope over the last K detections, causally.  K=2 is
                  the plain finite difference; larger K trades latency for noise.
  occluded frame  a frame where nobody is detected.  The stateless head fails
                  outright; the temporal one coasts on its last velocity.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

SEED = 0
GRID = 64  # cells per side
CELL = 0.5  # metres per cell
FPS = 10.0  # BEV frame rate
DT = 1.0 / FPS
FRAMES = 32  # 3.2 s of scene
N_OBJ = 8
SPEED_LO, SPEED_HI = 0.8, 3.2  # m/s -- near-field actors, pedestrian to cyclist
SIGMA_PRE = 0.25  # m -- detector position noise, before rasterisation
BLOB_SIGMA = 0.6  # m -- rendered occupancy kernel
OCC_THRESH = 0.5  # occupancy probability cut
WIN = 3  # cells of read-back window around a blob
KS = (2, 3, 4, 8)
OCC_FRAME = 24  # the frame every actor spends fully occluded


def scene(rng):
    """Ground-truth tracks ``(N_OBJ, FRAMES, 2)`` in metres, plus their velocities."""
    p0 = rng.uniform(-3.0, 3.0, size=(N_OBJ, 2))
    ang = rng.uniform(0.0, 2 * np.pi, size=N_OBJ)
    spd = rng.uniform(SPEED_LO, SPEED_HI, size=N_OBJ)
    vel = np.stack([spd * np.cos(ang), spd * np.sin(ang)], axis=1)
    t = np.arange(FRAMES) * DT
    return p0[:, None, :] + vel[:, None, :] * t[None, :, None], vel


def raster(q):
    """Paint one detection as occupancy, read the centroid back.  Metres in, metres out.

    Occupancy is a Gaussian blob thresholded at ``OCC_THRESH``; the read-back is the
    probability-weighted centroid of the surviving cells, so the grid quantises the
    position even though the detector's error was continuous.
    """
    gi = np.arange(int(GRID / 2 + q[1] / CELL) - WIN, int(GRID / 2 + q[1] / CELL) + WIN + 1) + 0.5
    gj = np.arange(int(GRID / 2 + q[0] / CELL) - WIN, int(GRID / 2 + q[0] / CELL) + WIN + 1) + 0.5
    gy, gx = np.meshgrid((gi - GRID / 2) * CELL, (gj - GRID / 2) * CELL, indexing="ij")
    p = np.exp(-((gx - q[0]) ** 2 + (gy - q[1]) ** 2) / (2 * BLOB_SIGMA**2))
    m = p >= OCC_THRESH
    if not m.any():
        return None
    w = p[m]
    return np.array([(gx[m] * w).sum() / w.sum(), (gy[m] * w).sum() / w.sum()])


def detections(track, noise, occlude: bool):
    """Per-frame per-actor read-back.  NaN means the actor was not seen at all."""
    det = np.full(track.shape, np.nan)
    for f in range(FRAMES):
        if occlude and f == OCC_FRAME:
            continue
        for i in range(N_OBJ):
            r = raster(track[i, f] + noise[i, f])
            if r is not None:
                det[i, f] = r
    return det


def slope(ts, xs):
    """Least-squares velocity over a window of detections.  K=2 is a difference."""
    t = ts - ts.mean()
    return (t[:, None] * xs).sum(0) / (t * t).sum()


def sweep(det, vel, ks):
    """Causal accuracy of the K-frame slope, averaged over actors and frames."""
    err = {k: [] for k in ks}
    for i in range(N_OBJ):
        for f in range(FRAMES):
            for k in ks:
                idx = np.flatnonzero(~np.isnan(det[i, : f + 1, 0]))[-k:]
                if len(idx) < k:
                    continue
                err[k].append(np.linalg.norm(slope(idx * DT, det[i, idx]) - vel[i]))
    return {k: float(np.mean(v)) for k, v in err.items()}


def main() -> int:
    rng = mapkit.rng(SEED)
    track, vel = scene(rng)
    # One noise tensor for both passes, so the clean and occluded runs differ in
    # exactly one thing: whether the frame-OCC_FRAME detection is there.
    noise = rng.normal(0.0, SIGMA_PRE, size=(N_OBJ, FRAMES, 2))

    clean = detections(track, noise, occlude=False)
    occl = detections(track, noise, occlude=True)

    mae = sweep(clean, vel, KS)
    best_k = min(mae, key=mae.get)

    seen = ~np.isnan(clean[:, :, 0])
    speed = np.linalg.norm(vel, axis=1)
    mean_speed = float(speed.mean())
    # Frame-stateless: no time axis, so the honest report is zero, and its MAE is the
    # scene's mean speed by construction.  That identity is the sanity check.
    single_frame_mae = float(np.repeat(speed[:, None], FRAMES, axis=1)[seen].mean())

    resid = (clean - track)[seen]
    obs_sigma = float(np.sqrt((resid**2).mean()))

    # Occluded frame: the stateless head has no detection, so it cannot answer at all.
    # The temporal head holds the velocity from the last best_k visible frames.
    blind = int(np.isnan(occl[:, OCC_FRAME, 0]).sum())
    hold = []
    for i in range(N_OBJ):
        idx = np.flatnonzero(~np.isnan(occl[i, :OCC_FRAME, 0]))[-best_k:]
        hold.append(np.linalg.norm(slope(idx * DT, occl[i, idx]) - vel[i]))
    occlusion_mae = float(np.mean(hold))

    rows = [
        {
            "K": k,
            "window_s": round((k - 1) * DT, 2),
            "velocity_mae_mps": round(mae[k], 4),
            "vs_single_frame": round(mae[k] / single_frame_mae, 3),
        }
        for k in KS
    ]

    metrics = {
        "grid_cells": GRID,
        "cell_m": CELL,
        "extent_m": round(GRID * CELL, 3),
        "fps": FPS,
        "frames": FRAMES,
        "n_objects": N_OBJ,
        "occlusion_frame": OCC_FRAME,
        "speed_lo_mps": SPEED_LO,
        "speed_hi_mps": SPEED_HI,
        "mean_speed_mps": round(mean_speed, 4),
        "detector_sigma_m": SIGMA_PRE,
        "obs_pos_sigma_m": round(obs_sigma, 4),
        # What the grid costs before any noise: a 0.5 m cell read back as a centroid
        # cannot be sharper than uniform quantisation.
        "quant_sigma_floor_m": round(CELL / np.sqrt(12), 4),
        # The noise on one position, differenced over one frame of baseline.  This is
        # why K=2 loses to guessing zero.
        "k2_noise_mps": round(obs_sigma * np.sqrt(2) / DT, 3),
        # Headline: a frame with no time axis can only say "zero", so it eats the whole
        # mean speed; a K-frame slope over the same frames eats a fraction.
        "single_frame_mae_mps": round(single_frame_mae, 4),
        "best_k": best_k,
        "finitediff_mae_mps": round(mae[best_k], 4),
        "error_reduction_vs_single_frame": round(single_frame_mae / mae[best_k], 3),
        "k2_over_single_frame": round(mae[2] / single_frame_mae, 3),
        "occlusion_single_frame_failure_rate": round(blind / N_OBJ, 4),
        "occlusion_temporal_mae_mps": round(occlusion_mae, 4),
        "occlusion_temporal_over_single_frame": round(occlusion_mae / single_frame_mae, 4),
    }

    path = mapkit.emit(pathlib.Path(__file__).resolve().parent, metrics, {"k_sweep": rows})
    print(
        f"single frame {single_frame_mae:.3f} m/s -> {best_k}-frame {mae[best_k]:.3f} m/s "
        f"| occluded: stateless {blind}/{N_OBJ} fail, temporal {occlusion_mae:.3f} m/s "
        f"-> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
