#!/usr/bin/env python
"""A benchmark cannot tell memorisation from understanding -- unless you shift it.

This is a *model* of the phenomenon, not a reproduction of it.  The design removes
every confound except one:

    both learners are lookup tables of the same construction and the same capacity.
    the only difference is what they index on.

One indexes on the **scene identifier** -- constant within a task, therefore a
perfect predictor on a fixed benchmark, but carrying no semantics and not existing
in the world.  The other indexes on the object's **colour**, which is what the task
actually depends on.

The result the block is built to show is not that the probe wins.  It is that on the
standard benchmark **the two are indistinguishable**, and the probe is also the
cheaper of the two.  A leaderboard built on a fixed task set cannot prefer the one
that learned something.

Reference: LIBERO-PRO reports that perturbing standard LIBERO tasks collapses >90%
of models, and a parameter-free-lookup probe has been shown to reach SOTA scores.
Those citations are marked !CITE in the README; every number below is this block's.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import mapkit  # noqa: E402
import numpy as np  # noqa: E402

N_COLORS = 6  # action classes == object colours, i.e. the actual semantics
N_TASKS = 24  # benchmark tasks; each colour appears in 4 of them
N_TRAIN = 60  # demonstrations per task
N_TEST = 40  # rollouts per task
LABEL_NOISE = 0.10  # imperfect demonstrations


def build(rng):
    """Scene id and colour are perfectly confounded inside each task."""
    scenes, colors, labels = [], [], []
    for task in range(N_TASKS):
        color = task % N_COLORS
        for _ in range(N_TRAIN):
            scenes.append(task)
            colors.append(color)
            labels.append(color if rng.random() > LABEL_NOISE else rng.integers(N_COLORS))
    return np.array(scenes), np.array(colors), np.array(labels)


def table(keys, labels) -> dict[int, int]:
    """Identical learner for every key function -- same capacity, same data."""
    buckets: dict[int, Counter] = {}
    for k, y in zip(keys, labels, strict=True):
        buckets.setdefault(int(k), Counter())[int(y)] += 1
    return {k: c.most_common(1)[0][0] for k, c in buckets.items()}


def accuracy(lookup, keys, truth) -> float:
    return float(np.mean(np.array([lookup.get(int(k), 0) for k in keys]) == np.array(truth)))


def main() -> int:
    rng = mapkit.rng(0)
    scenes, colors, labels = build(rng)

    probe = table(scenes, labels)  # keyed on scene identity -- the shortcut
    generalist = table(colors, labels)  # keyed on colour -- the semantics

    # --- A. the standard benchmark: the tasks you trained on ---------------
    std_scene = np.repeat(np.arange(N_TASKS), N_TEST)
    std_color = std_scene % N_COLORS
    std_truth = std_color.copy()
    probe_std = accuracy(probe, std_scene, std_truth)
    gen_std = accuracy(generalist, std_color, std_truth)

    # --- B. scene shift: new contexts, semantics untouched -----------------
    shift_scene = 10000 + np.repeat(np.arange(N_TASKS), N_TEST)
    shift_color = np.repeat(np.arange(N_TASKS), N_TEST) % N_COLORS
    probe_shift = accuracy(probe, shift_scene, shift_color)
    gen_shift = accuracy(generalist, shift_color, shift_color)

    # --- C. genuinely new colours: neither can do this ---------------------
    # Unseen colours with randomly assigned true actions.  Both learners fall
    # back to their default for an unseen key, so both should land at chance.
    # Included so the block does not overclaim what the generalist generalises to.
    n_novel = 600
    novel_colors = N_COLORS + 1 + rng.integers(0, 1000, n_novel)
    novel_truth = rng.integers(0, N_COLORS, n_novel)
    probe_new = accuracy(probe, novel_colors, novel_truth)
    gen_new = accuracy(generalist, novel_colors, novel_truth)

    metrics = {
        "n_colors": N_COLORS,
        "n_tasks": N_TASKS,
        "tasks_per_color": N_TASKS // N_COLORS,
        "n_train_per_task": N_TRAIN,
        "n_test_per_task": N_TEST,
        "label_noise": LABEL_NOISE,
        "probe_table_entries": len(probe),
        "generalist_table_entries": len(generalist),
        # The headline: on the standard split these are the same number.
        "probe_std_acc": round(probe_std, 4),
        "generalist_std_acc": round(gen_std, 4),
        "std_gap_pp": round((probe_std - gen_std) * 100, 2),
        # ...and here they are not.
        "probe_scene_shift_acc": round(probe_shift, 4),
        "generalist_scene_shift_acc": round(gen_shift, 4),
        "probe_scene_shift_drop_pp": round((probe_std - probe_shift) * 100, 2),
        "generalist_scene_shift_drop_pp": round((gen_std - gen_shift) * 100, 2),
        "probe_new_color_acc": round(probe_new, 4),
        "generalist_new_color_acc": round(gen_new, 4),
        "chance_acc": round(1.0 / N_COLORS, 4),
    }

    rows = [
        {"split": "标准基准", "probe": round(probe_std, 4), "generalist": round(gen_std, 4)},
        {"split": "场景替换", "probe": round(probe_shift, 4), "generalist": round(gen_shift, 4)},
        {"split": "目标替换", "probe": round(probe_new, 4), "generalist": round(gen_new, 4)},
    ]

    path = mapkit.emit(pathlib.Path(__file__).resolve().parent, metrics, {"splits": rows})
    print(
        f"standard: probe {probe_std:.3f} == generalist {gen_std:.3f} | "
        f"scene-shift: {probe_shift:.3f} vs {gen_shift:.3f} -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
