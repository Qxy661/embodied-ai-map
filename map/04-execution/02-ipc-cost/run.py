#!/usr/bin/env python
"""What it costs to move data across the seams of a split robot pipeline.

Ring 04-01 showed that a general-purpose OS cannot promise a deadline.  This block
takes the next step.  Once you stop pretending that one process can do everything,
the pipeline falls apart into camera / perception / planning / servo processes --
for crash isolation, because one of them is Python and another is C++, and because
the servo loop needs a real-time scheduling class that the inference process must
not be allowed to steal.  Every split is a seam, and every seam is a place where
bytes have to cross.

The block prices that crossing, and it is **entirely arithmetic**.  Nothing here is
timed, so nothing here is ``volatile``: the same inputs produce byte-identical
metrics on every machine, which is what lets a README quote a sweep table at all.

That is a deliberate division of labour rather than an omission.  04-01 is the
ring's live measurement, and its lesson is that a measured millisecond does not
transfer between hosts.  What transfers is the *count*: 6.2 MB per frame, four
copies to drag it through a pipe, twenty-two microseconds to send the boxes
instead.  So 04-01 says the deadline cannot be promised and this block says the
bytes are not free either -- and the second claim is the one you can hand to a
colleague with a different laptop.

The copy counts in ``COPY_COUNTS`` are the only modelling assumption in the file.
They are declared next to what each one assumes, and the README's 边界 section says
where they are a model of a transport rather than a measurement of one.
"""

from __future__ import annotations

import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

# --- the pipeline being priced ------------------------------------------------
FRAME_W, FRAME_H, FRAME_C = 1920, 1080, 3
FPS = 30.0
CTRL_HZ = 1000.0
N_BOXES = 100
BOX_FIELDS = 7  # x, y, z, dx, dy, dz, yaw -- the 7-DoF box convention
FEATURE_DIM = 1024
N_JOINTS = 32
F32 = 4
# Deliberately conservative.  A real 1080p30 H.264 stream is far smaller than
# this, so the compressed row below is the pessimistic end of the option.
H264_RATIO = 50

# Full-payload copies charged to each transport, per message.  This is a model of
# each design, not a timing: the point is that the count does not depend on how
# fast the machine is.
COPY_COUNTS = (
    ("pipe_pickle", 4, "序列化、内核写、内核读、反序列化"),
    ("pipe_raw", 2, "内核写、内核读"),
    ("shm_handle", 1, "生产者写进共享缓冲区；消费者就地读，不再拷贝"),
    ("loaned", 0, "缓冲区由生产者写满并移交所有权"),
)

# Nameplate link budgets, decimal megabytes per second.  Constants of the
# standard, not measurements of this host.
LINKS = (
    ("gbe", 125.0, "千兆以太网"),
    ("usb3", 625.0, "USB 3.2 Gen1"),
    ("ten_gbe", 1250.0, "万兆以太网"),
)
# Dual-channel DDR4-3200 at a practical half of its nameplate.  Only used to show
# that memory bandwidth is *not* what this block is up against.
DDR_PRACTICAL_MBPS = 25000.0

RESOLUTIONS = (
    ("VGA", 640, 480),
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("4K", 3840, 2160),
)
SWEEP_FPS = (10, 30, 60)


def mb_per_s(bytes_per_s: float) -> float:
    """Decimal MB/s, rounded to the precision the README is allowed to write."""
    return round(bytes_per_s / 1e6, 2)


def main() -> int:
    # A real frame, seeded, so the nbytes and the pickle size below are computed
    # rather than asserted.  pickle of a contiguous array is a memcpy plus a few
    # bytes of header, which is the honest thing to say about it: it does not
    # inflate the payload, it makes it cross the bus twice more.
    frame = mapkit.rng(0).integers(0, 256, size=(FRAME_H, FRAME_W, FRAME_C), dtype=np.uint8)
    frame_bytes = int(frame.nbytes)
    wire_bytes = len(pickle.dumps(frame))
    pickle_overhead_bytes = wire_bytes - frame_bytes

    detection_bytes = N_BOXES * BOX_FIELDS * F32
    feature_bytes = FEATURE_DIM * F32
    ctrl_bytes = N_JOINTS * F32
    h264_bytes = int(round(frame_bytes / H264_RATIO))

    period_ms = round(1000.0 / FPS, 2)
    frame_per_s = frame_bytes * FPS
    detection_per_s = detection_bytes * FPS
    feature_per_s = feature_bytes * FPS
    ctrl_per_s = ctrl_bytes * CTRL_HZ
    h264_per_s = h264_bytes * FPS

    # --- transports -----------------------------------------------------------
    transports = []
    for name, copies, note in COPY_COUNTS:
        transports.append(
            {
                "transport": name,
                "copies_per_message": copies,
                "note": note,
                "mem_bw_mb_per_s": mb_per_s(frame_per_s * copies),
                # What actually has to cross the wire: the payload once per hop.
                # Copies are memory traffic, not link traffic; keeping the two
                # apart is what stops this table from double-counting.
                "link_mb_per_s": mb_per_s(frame_per_s),
            }
        )
    by_transport = {t["transport"]: t for t in transports}

    # --- link budgets ---------------------------------------------------------
    links = []
    for name, mbps, note in LINKS:
        bps = mbps * 1e6
        links.append(
            {
                "link": name,
                "note": note,
                "mb_per_s": mbps,
                "frame_frac": round(frame_per_s / bps, 4),
                "detection_frac": round(detection_per_s / bps, 6),
                "frame_ms": round(frame_bytes / bps * 1e3, 2),
                "h264_ms": round(h264_bytes / bps * 1e3, 3),
                "detection_ms": round(detection_bytes / bps * 1e3, 6),
                "detection_us": round(detection_bytes / bps * 1e6, 1),
            }
        )
    by_link = {x["link"]: x for x in links}
    gbe = by_link["gbe"]

    # --- resolution x frame rate ---------------------------------------------
    sweep = []
    for label, w, h in RESOLUTIONS:
        for hz in SWEEP_FPS:
            bps = w * h * FRAME_C * hz
            sweep.append(
                {
                    "resolution": label,
                    "w": w,
                    "h": h,
                    "mpx": round(w * h / 1e6, 2),
                    "fps": hz,
                    "mb_per_s": mb_per_s(bps),
                    "gbe_frac": round(bps / (gbe["mb_per_s"] * 1e6), 3),
                    "fits_gbe": bool(bps <= gbe["mb_per_s"] * 1e6),
                }
            )

    # The two thresholds that make the sweep actionable, solved rather than read
    # off the grid.
    max_mpx_gbe_at_30hz = round(gbe["mb_per_s"] * 1e6 / (FPS * FRAME_C) / 1e6, 2)
    max_fps_1080p_gbe = round(gbe["mb_per_s"] * 1e6 / frame_bytes, 2)

    # --- the seams ------------------------------------------------------------
    # Three process boundaries: camera -> perception -> planning -> servo.  Sending
    # the image through all of them is what a split with no discipline costs.
    n_hops = 3
    hops = [
        {
            "hop": "camera -> perception",
            "payload": "1080p RGB",
            "bytes": frame_bytes,
            "hz": FPS,
            "mb_per_s": mb_per_s(frame_per_s),
        },
        {
            "hop": "perception -> planning",
            "payload": "detections",
            "bytes": detection_bytes,
            "hz": FPS,
            "mb_per_s": mb_per_s(detection_per_s),
        },
        {
            "hop": "planning -> servo",
            "payload": "joint command",
            "bytes": ctrl_bytes,
            "hz": CTRL_HZ,
            "mb_per_s": mb_per_s(ctrl_per_s),
        },
    ]
    disciplined = frame_per_s + detection_per_s + ctrl_per_s
    naive = frame_per_s * n_hops

    metrics = {
        # --- the pipeline -----------------------------------------------------
        "frame_w": FRAME_W,
        "frame_h": FRAME_H,
        "frame_c": FRAME_C,
        "frame_bytes": frame_bytes,
        "frame_mb": round(frame_bytes / 1e6, 2),
        "frame_mib": round(frame_bytes / 2**20, 2),
        "fps": FPS,
        "frame_period_ms": period_ms,
        "ctrl_hz": CTRL_HZ,
        "n_joints": N_JOINTS,
        "n_boxes": N_BOXES,
        "box_fields": BOX_FIELDS,
        "feature_dim": FEATURE_DIM,
        "n_hops": n_hops,
        # --- what pickle actually does to a numpy array ------------------------
        "pickle_wire_bytes": wire_bytes,
        "pickle_overhead_bytes": pickle_overhead_bytes,
        # --- rates ------------------------------------------------------------
        "frame_mb_per_s": mb_per_s(frame_per_s),
        "h264_ratio_assumed": H264_RATIO,
        "h264_bytes": h264_bytes,
        "h264_mb_per_s": mb_per_s(h264_per_s),
        "detection_bytes": detection_bytes,
        "detection_mb_per_s": mb_per_s(detection_per_s),
        "feature_bytes": feature_bytes,
        "feature_mb_per_s": mb_per_s(feature_per_s),
        "ctrl_bytes": ctrl_bytes,
        "ctrl_mb_per_s": mb_per_s(ctrl_per_s),
        # The whole argument for "send boxes, not pictures", as one ratio.
        "ratio_frame_to_detection": round(frame_per_s / detection_per_s, 1),
        "ratio_frame_to_feature": round(frame_per_s / feature_per_s, 1),
        "ratio_frame_to_ctrl": round(frame_per_s / ctrl_per_s, 1),
        "ratio_ctrl_hz_to_cam_hz": round(CTRL_HZ / FPS, 2),
        # --- transports -------------------------------------------------------
        "copies_pipe_pickle": by_transport["pipe_pickle"]["copies_per_message"],
        "copies_pipe_raw": by_transport["pipe_raw"]["copies_per_message"],
        "copies_shm_handle": by_transport["shm_handle"]["copies_per_message"],
        "copies_loaned": by_transport["loaned"]["copies_per_message"],
        "mem_bw_pipe_pickle_mb_per_s": by_transport["pipe_pickle"]["mem_bw_mb_per_s"],
        "mem_bw_pipe_raw_mb_per_s": by_transport["pipe_raw"]["mem_bw_mb_per_s"],
        "mem_bw_shm_mb_per_s": by_transport["shm_handle"]["mem_bw_mb_per_s"],
        "mem_bw_loaned_mb_per_s": by_transport["loaned"]["mem_bw_mb_per_s"],
        "ddr_practical_mb_per_s": DDR_PRACTICAL_MBPS,
        # The honest headline: even the naive path is nowhere near memory-bound.
        "mem_frac_of_ddr_pipe_pickle": round(
            frame_per_s * COPY_COUNTS[0][1] / (DDR_PRACTICAL_MBPS * 1e6), 4
        ),
        # --- links ------------------------------------------------------------
        "link_gbe_mb_per_s": gbe["mb_per_s"],
        "link_usb3_mb_per_s": by_link["usb3"]["mb_per_s"],
        "link_ten_gbe_mb_per_s": by_link["ten_gbe"]["mb_per_s"],
        "frame_frac_of_gbe": gbe["frame_frac"],
        "frame_frac_of_ten_gbe": by_link["ten_gbe"]["frame_frac"],
        "detection_frac_of_gbe": gbe["detection_frac"],
        "frame_ms_on_gbe": gbe["frame_ms"],
        "frame_ms_on_ten_gbe": by_link["ten_gbe"]["frame_ms"],
        "h264_ms_on_gbe": gbe["h264_ms"],
        "detection_ms_on_gbe": gbe["detection_ms"],
        "detection_us_on_gbe": gbe["detection_us"],
        "gbe_frame_over_period": round(gbe["frame_ms"] / period_ms, 2),
        "ten_gbe_frame_over_period": round(by_link["ten_gbe"]["frame_ms"] / period_ms, 3),
        # --- thresholds -------------------------------------------------------
        "max_mpx_gbe_at_30hz": max_mpx_gbe_at_30hz,
        "max_fps_1080p_gbe": max_fps_1080p_gbe,
        # --- the seams --------------------------------------------------------
        "naive_pipeline_mb_per_s": mb_per_s(naive),
        "disciplined_pipeline_mb_per_s": mb_per_s(disciplined),
        "naive_over_disciplined": round(naive / disciplined, 2),
    }

    tables = {
        "payloads": [
            {
                "payload": "1080p RGB frame",
                "bytes": frame_bytes,
                "hz": FPS,
                "mb_per_s": mb_per_s(frame_per_s),
            },
            {
                "payload": "H.264 stream",
                "bytes": h264_bytes,
                "hz": FPS,
                "mb_per_s": mb_per_s(h264_per_s),
            },
            {
                "payload": "100 boxes x 7 f32",
                "bytes": detection_bytes,
                "hz": FPS,
                "mb_per_s": mb_per_s(detection_per_s),
            },
            {
                "payload": "feature 1024 f32",
                "bytes": feature_bytes,
                "hz": FPS,
                "mb_per_s": mb_per_s(feature_per_s),
            },
            {
                "payload": "joint command 32 f32",
                "bytes": ctrl_bytes,
                "hz": CTRL_HZ,
                "mb_per_s": mb_per_s(ctrl_per_s),
            },
        ],
        "transports": transports,
        "links": links,
        "sweep": sweep,
        "hops": hops,
    }

    path = mapkit.emit(
        pathlib.Path(__file__).resolve().parent,
        metrics,
        tables,
        notes=(
            "Entirely arithmetic, no timing: deterministic by construction and "
            "checked exactly. The copy counts in COPY_COUNTS are the one modelling "
            "assumption; they describe each transport's design, not this host. "
            "Link rates are nameplate constants of the standards. 04-01 owns the "
            "live measurement in this ring."
        ),
    )
    print(
        f"{FRAME_W}x{FRAME_H}x{FRAME_C} @ {FPS:.0f} Hz = {metrics['frame_mb_per_s']:.2f} MB/s, "
        f"x{metrics['copies_pipe_pickle']} copies through a pipe = "
        f"{metrics['mem_bw_pipe_pickle_mb_per_s']:.2f} MB/s | boxes are "
        f"{metrics['ratio_frame_to_detection']:.1f}x cheaper | on GbE the frame takes "
        f"{metrics['frame_ms_on_gbe']:.2f} ms of a {period_ms:.2f} ms period "
        f"({metrics['gbe_frame_over_period']:.2f}x) -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
