#!/usr/bin/env python
"""Lift-Splat-Shoot, reduced to the one geometric claim it actually makes.

LSS is usually presented as an architecture.  It is really one claim about *where* to
fuse cameras: on the ground plane, not in the image plane.  This block builds the
smallest scene in which that claim is testable and measures the numbers behind it.

Depth is never invented: every pixel's true depth comes from ray casting against the
ground and the poles, so unprojecting at that depth and reprojecting lands back on the
same pixel.  The block asserts this, because without it nothing downstream means
anything.

Pipeline: lift (unproject each pixel into one 3D point per depth bin, weighted by a
predicted depth distribution) -> splat (accumulate into a BEV grid; this *is* voxel
pooling).  Shoot is a learned head that adds no geometry, so the block stops there.
"""

from __future__ import annotations

import pathlib
import sys
from collections import namedtuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

W, H = 160, 120
FX = FY = 100.0
CX, CY = W / 2.0, H / 2.0
N_BINS, D_MIN, D_MAX = 32, 1.0, 20.0
SIGMA_P = 0.10  # width of the predicted depth distribution, metres (~1/3 bin)
POLE_R, POLE_H = 0.15, 1.8
POLES = np.array([[3.5, 0.0], [6.0, 2.2], [6.5, -2.4], [9.0, 0.9], [9.0, -1.6]])
CAMS = [(0.0, 0.0, 0.0), (0.0, 0.4, 30.0), (0.0, -0.4, -30.0)]  # x, y, yaw deg
CAM_Z, PITCH_DEG = 1.4, 20.0
PAIR, PAIR_POLE = (0, 1), 1  # front + left-front, and the pole both of them see
NOISE_ABS, NOISE_REL = 0.10, 0.010
EXTENT = 10.0  # grid spans [-10, 10]^2 -> 20 m x 20 m
CELLS = [0.05, 0.1, 0.2, 0.4, 0.8]
TAU = 1.0  # a cell is occupied at >= one pixel-worth of accumulated evidence
TAUS = np.arange(0.2, 40.0, 0.2)  # swept, for the "best operating point" columns
SEED = 0

EDGES = np.geomspace(D_MIN, D_MAX, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
ONEHOT = np.eye(N_BINS)
UV = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)

# C, R: camera pose.  t, valid, which: ray-cast depth, mask, pole index (-1 = not a pole).
# feat, pdist: the pixel feature and its predicted depth distribution.
Frame = namedtuple("Frame", "C R d t valid which feat pdist")


def cam_rot(yaw_deg: float, pitch_deg: float = PITCH_DEG) -> np.ndarray:
    """World->camera rotation. Rows are the camera axes (right, down, forward)."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    return np.stack(
        [
            [np.sin(y), -np.cos(y), 0.0],
            [-np.sin(p) * np.cos(y), -np.sin(p) * np.sin(y), -np.cos(p)],
            [np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), -np.sin(p)],
        ]
    )


def rays(cam):
    x, y, yaw = cam
    C, R = np.array([x, y, CAM_Z]), cam_rot(yaw)
    d = np.stack([(UV[0] - CX) / FX, (UV[1] - CY) / FY, np.ones_like(UV[0])], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return C, R, d @ R  # (H, W, 3) unit rays in world


def render(C, d):
    """Ray cast: distance to the first surface, validity, and which pole was hit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[..., 2] < -1e-9, -C[2] / d[..., 2], np.inf)
    which = np.full(d.shape[:2], -1, np.int64)
    for k, (px, py) in enumerate(POLES):  # finite vertical cylinder
        ox, oy = C[0] - px, C[1] - py
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2.0 * (ox * d[..., 0] + oy * d[..., 1])
        disc = b * b - 4.0 * a * (ox * ox + oy * oy - POLE_R**2)
        ok = disc > 0.0
        tp = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * a)
        z = C[2] + tp * d[..., 2]
        win = ok & (tp > 1e-6) & (z >= 0.0) & (z <= POLE_H) & (tp < t)
        t, which = np.where(win, tp, t), np.where(win, k, which)
    valid = np.isfinite(t)
    return np.where(valid, t, 0.0), valid, which


def project(C, R, P):
    q = R @ (P - C)
    return FX * q[0] / q[2] + CX, FY * q[1] / q[2] + CY


def soft_dist(depth):
    z = -((CENTERS[None, None, :] - depth[..., None]) ** 2) / (2.0 * SIGMA_P**2)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def lift(C, d, cell):
    """Unproject every (pixel, depth bin) pair to a flat BEV cell index.

    This is the expensive half and it depends only on the geometry, so soft and hard
    depth share it -- that is the whole point of factoring it out.
    """
    n = int(round(2 * EXTENT / cell))
    pts = C[:2] + d[..., None, :2] * CENTERS[None, None, :, None]  # (H, W, D, 2)
    ij = np.floor((pts + EXTENT) / cell).astype(np.int64)
    inside = np.all((ij >= 0) & (ij < n), axis=-1)
    flat = np.where(inside, ij[..., 0] * n + ij[..., 1], -1).ravel()
    keep = flat >= 0
    return n, flat[keep], keep


def splat(geom, weight, feat):
    """Voxel pooling: accumulate weighted points into a BEV grid."""
    n, flat, keep = geom
    w = (weight * feat[..., None]).ravel()[keep]
    return np.bincount(flat, weights=w, minlength=n * n).reshape(n, n)


def fused(geoms, frames, hard):
    """Three cameras splatted into one BEV map.  ``hard`` = argmax instead of soft."""
    total = 0.0
    for geom, f in zip(geoms, frames, strict=True):
        weight = f.pdist
        if hard:
            weight = ONEHOT[f.pdist.argmax(-1)] * (f.pdist.sum(-1, keepdims=True) > 0)
        total = total + splat(geom, weight, f.feat)
    return total


def truth_grid(cell):
    """Cells whose centre falls inside a pole footprint dilated by half a cell."""
    n = int(round(2 * EXTENT / cell))
    c = -EXTENT + (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(c, c, indexing="ij")
    dil = POLE_R + 0.5 * cell * np.sqrt(2.0)
    occ = np.zeros((n, n), bool)
    for px, py in POLES:
        occ |= (gx - px) ** 2 + (gy - py) ** 2 <= dil * dil
    return occ


def iou(heat, truth, tau=TAU):
    pred = heat >= tau
    return float(np.count_nonzero(pred & truth) / np.count_nonzero(pred | truth))


def best_iou(heat, truth):
    return float(max(iou(heat, truth, t) for t in TAUS))


def centroid(heat, cell, target, half=1.0):
    """Mass centroid near ``target``, in a window tight enough to exclude neighbours."""
    n = heat.shape[0]
    c = -EXTENT + (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(c, c, indexing="ij")
    m = (np.abs(gx - target[0]) <= half) & (np.abs(gy - target[1]) <= half)
    w = heat[m]
    return None if w.sum() <= 0.0 else np.array([(w * gx[m]).sum(), (w * gy[m]).sum()]) / w.sum()


def localisation(geoms, frames, cell):
    """Mean error of soft / hard vs the true point cloud, and the near-surface bias.

    The reference is the centroid of the *observed* 3D points of that pole in that
    camera -- the near surface of the cylinder, not its axis.  Against the axis, a
    0.15 m observation bias that no depth head can remove buries the comparison.
    """
    errs, bias = [], []
    for f, geom in zip(frames, geoms, strict=True):
        one = ONEHOT[f.pdist.argmax(-1)] * (f.pdist.sum(-1, keepdims=True) > 0)
        h_s, h_h = splat(geom, f.pdist, f.feat), splat(geom, one, f.feat)
        surface = f.C[:2] + f.d[..., :2] * f.t[..., None]
        for k in range(len(POLES)):
            m = (f.which == k) & f.valid
            if not m.any():
                continue
            ref = surface[m].mean(0)
            cs, ch = centroid(h_s, cell, ref), centroid(h_h, cell, ref)
            if cs is None or ch is None:
                continue
            errs.append([np.linalg.norm(cs - ref), np.linalg.norm(ch - ref)])
            bias.append(np.linalg.norm(ref - POLES[k]))
    e = np.array(errs)
    return float(e[:, 0].mean()), float(e[:, 1].mean()), float(np.mean(bias)), len(e)


def main() -> int:
    rng = mapkit.rng(SEED)
    frames = []
    for cam in CAMS:
        C, R, d = rays(cam)
        t, valid, which = render(C, d)
        noisy = np.clip(
            t + rng.normal(0.0, 1.0, t.shape) * (NOISE_ABS + NOISE_REL * t),
            D_MIN + 1e-3,
            D_MAX - 1e-3,
        )
        frames.append(
            Frame(
                C,
                R,
                d,
                t,
                valid,
                which,
                (which >= 0).astype(float),
                soft_dist(noisy) * valid[..., None],
            )
        )
    pole_px = [f.which >= 0 for f in frames]
    geoms = {c: [lift(f.C, f.d, c) for f in frames] for c in CELLS}

    # Geometry self-consistency: unprojecting at the true depth and reprojecting must
    # land back on the same pixel.  Everything below rests on this being ~0.
    reproj = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for f in frames:
            q = (f.d * f.t[..., None]) @ f.R.T  # point relative to C, in camera frame
            du = FX * q[..., 0] / q[..., 2] + CX - UV[0]
            dv = FY * q[..., 1] / q[..., 2] + CY - UV[1]
            reproj = max(reproj, float(np.max(np.hypot(du, dv)[f.valid])))
    assert reproj < 1e-6, f"reprojection error {reproj}"

    # -- (a) soft depth vs hard depth, three cameras fused ------------------------
    cell = 0.1
    truth = truth_grid(cell)
    heat_soft, heat_hard = fused(geoms[cell], frames, False), fused(geoms[cell], frames, True)
    loc_soft, loc_hard, surf_bias, n_pairs = localisation(geoms[cell], frames, cell)

    # Depth error along the ray, pole pixels only: mean of the distribution vs its mode.
    maes = np.array(
        [
            [
                np.abs((f.pdist * CENTERS).sum(-1)[m] - f.t[m]).mean(),
                np.abs(CENTERS[f.pdist.argmax(-1)][m] - f.t[m]).mean(),
            ]
            for f, m in zip(frames, pole_px, strict=True)
        ]
    )

    # -- (b) the same pole, seen by two cameras -----------------------------------
    fa, fb = frames[PAIR[0]], frames[PAIR[1]]
    P = np.array([POLES[PAIR_POLE][0], POLES[PAIR_POLE][1], POLE_H / 2.0])
    ua, va = project(fa.C, fa.R, P)
    ub, vb = project(fb.C, fb.R, P)
    img_dist = float(np.hypot(ua - ub, va - vb))
    ca = centroid(splat(geoms[cell][PAIR[0]], fa.pdist, fa.feat), cell, POLES[PAIR_POLE])
    cb = centroid(splat(geoms[cell][PAIR[1]], fb.pdist, fb.feat), cell, POLES[PAIR_POLE])
    bev_dist = float(np.hypot(*(ca - cb)))
    ia = tuple(int(np.floor((v + EXTENT) / cell)) for v in ca)
    ib = tuple(int(np.floor((v + EXTENT) / cell)) for v in cb)

    # -- (c) BEV resolution sweep --------------------------------------------------
    sweep = []
    for c in CELLS:
        t_grid, n = truth_grid(c), int(round(2 * EXTENT / c))
        ls, lh, _, _ = localisation(geoms[c], frames, c)
        sweep.append(
            {
                "cell_m": c,
                "cells_per_side": n,
                "n_cells": n * n,
                "truth_cells": int(t_grid.sum()),
                "truth_area_m2": round(t_grid.sum() * c * c, 4),
                "iou_soft": round(iou(fused(geoms[c], frames, False), t_grid), 4),
                "iou_hard": round(iou(fused(geoms[c], frames, True), t_grid), 4),
                "loc_soft_m": round(ls, 4),
                "loc_hard_m": round(lh, 4),
            }
        )
    best = max(sweep, key=lambda r: r["iou_soft"])
    sharpest = min(sweep, key=lambda r: r["loc_soft_m"])

    n_pole_px = int(sum(int(m.sum()) for m in pole_px))
    d_soft, d_hard = round(float(maes[:, 0].mean()), 4), round(float(maes[:, 1].mean()), 4)
    i_soft, i_hard = round(iou(heat_soft, truth), 4), round(iou(heat_hard, truth), 4)
    metrics = {
        # scene / depth head / grid -- setup, echoed so the README can cite it
        "n_cameras": len(CAMS),
        "n_poles": len(POLES),
        "grid_extent_m": 2 * EXTENT,
        "pole_radius_m": POLE_R,
        "pole_height_m": POLE_H,
        "pole_diameter_m": 2 * POLE_R,
        "pole_footprint_m2": round(float(np.pi * POLE_R**2), 4),
        "poles_footprint_m2": round(len(POLES) * float(np.pi * POLE_R**2), 4),
        "n_bins": N_BINS,
        "d_min_m": D_MIN,
        "d_max_m": D_MAX,
        "sigma_p_m": SIGMA_P,
        "cell_m": cell,
        "noise_abs_m": NOISE_ABS,
        "noise_rel": NOISE_REL,
        "tau_evidence": TAU,
        "cells_per_side": int(round(2 * EXTENT / cell)),
        "n_valid_pixels": int(sum(int(f.valid.sum()) for f in frames)),
        "n_pole_pixels": n_pole_px,
        "n_points_lifted": n_pole_px * N_BINS,
        "reproj_err_max_px": reproj,
        # (a) soft vs hard depth
        "depth_mae_soft_m": d_soft,
        "depth_mae_hard_m": d_hard,
        "depth_mae_gain_pct": round(100.0 * (1.0 - d_soft / d_hard), 1),
        "iou_soft": i_soft,
        "iou_hard": i_hard,
        "iou_gain_pct": round(100.0 * (i_soft / i_hard - 1.0), 1),
        "iou_soft_best": round(best_iou(heat_soft, truth), 4),
        "iou_hard_best": round(best_iou(heat_hard, truth), 4),
        "loc_soft_m": round(loc_soft, 4),
        "loc_hard_m": round(loc_hard, 4),
        "loc_gain_pct": round(100.0 * (1.0 - loc_soft / loc_hard), 1),
        "loc_pairs": n_pairs,
        "surface_bias_m": round(surf_bias, 4),
        # (b) one pole, two cameras
        "pair_u_front_px": round(float(ua), 2),
        "pair_v_front_px": round(float(va), 2),
        "pair_u_left_px": round(float(ub), 2),
        "pair_v_left_px": round(float(vb), 2),
        "pair_bev_front_x_m": round(float(ca[0]), 3),
        "pair_bev_front_y_m": round(float(ca[1]), 3),
        "pair_bev_left_x_m": round(float(cb[0]), 3),
        "pair_bev_left_y_m": round(float(cb[1]), 3),
        "pair_true_x_m": POLES[PAIR_POLE][0],
        "pair_true_y_m": POLES[PAIR_POLE][1],
        "pair_img_dist_px": round(img_dist, 2),
        "pair_bev_dist_m": round(bev_dist, 4),
        "pair_ratio": round(img_dist / bev_dist, 1),
        "pair_same_cell": int(ia == ib),
        "pair_cell_gap": int(max(abs(ia[0] - ib[0]), abs(ia[1] - ib[1]))),
        # (c) resolution sweep
        "best_cell_m": best["cell_m"],
        "best_cell_iou_soft": best["iou_soft"],
        "sharpest_cell_m": sharpest["cell_m"],
    }
    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        {"grid_sweep": sweep},
        seed=SEED,
        notes="Ray-cast ground truth. Soft depth weights every bin by the predicted "
        "distribution; hard depth drops all evidence on the argmax bin. Same occupancy "
        "threshold, same centroid window, same sweep for both.",
    )
    print(
        f"IoU soft {i_soft:.4f} vs hard {i_hard:.4f} | same pole: {img_dist:.1f} px apart in "
        f"image, {bev_dist:.3f} m in BEV, same cell {bool(ia == ib)} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
