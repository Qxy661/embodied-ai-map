"""The number checker is the mechanism the whole repository rests on.

If it silently stops biting, every block's prose is free to drift from its run and
the project degrades into a blog post with a code folder bolted on.  So it gets a
test that deliberately breaks a number and asserts the checker notices.

These tests build throwaway blocks in a temp directory rather than touching `map/`,
so they stay fast and never depend on which blocks currently exist.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import check_numbers as cn  # noqa: E402


def make_block(
    tmp_path: pathlib.Path,
    readme: str,
    metrics: dict,
    *,
    literals: list | None = None,
    volatile: bool = False,
    citable: list | None = None,
    tables: dict | None = None,
) -> pathlib.Path:
    block = tmp_path / "block"
    block.mkdir(exist_ok=True)
    (block / "README.md").write_text(readme, encoding="utf-8")
    (block / "run.py").write_text("# stand-in\n", encoding="utf-8")
    (block / "results.json").write_text(
        json.dumps({"block": "block", "metrics": metrics, "tables": tables or {}}),
        encoding="utf-8",
    )
    meta: dict = {}
    if literals:
        meta["literals"] = literals
    if volatile:
        meta["volatile"] = True
    if citable:
        meta["citable"] = citable
    (block / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return block


# --- the invariant: prose must trace to a run -------------------------------


def test_matching_number_passes(tmp_path):
    block = make_block(tmp_path, "准确率是 0.8314。\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert errors == []


def test_rounded_number_passes(tmp_path):
    """Prose rounds; results hold full precision.  That has to be allowed."""
    block = make_block(tmp_path, "准确率是 0.83。\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert errors == []


def test_fraction_written_as_percent_passes(tmp_path):
    block = make_block(tmp_path, "准确率是 83.14%。\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert errors == []


def test_invented_number_fails(tmp_path):
    """The whole point.  A number no run produced must be caught."""
    block = make_block(tmp_path, "准确率是 0.9999。\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert len(errors) == 1
    assert "0.9999" in errors[0]


def test_cite_marker_exempts_a_line(tmp_path):
    """Quoted-from-a-paper numbers are allowed, but only when marked."""
    block = make_block(tmp_path, "论文报告 99.2 分 !CITE\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert errors == []


def test_unmarked_paper_number_fails(tmp_path):
    block = make_block(tmp_path, "论文报告 99.2 分\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert len(errors) == 1


def test_literal_is_allowed(tmp_path):
    block = make_block(tmp_path, "分块长度 H 取 50。\n", {"acc": 0.8314}, literals=[50])
    errors, _ = cn.check_block(block)
    assert errors == []


def test_number_inside_identifier_is_not_matched(tmp_path):
    """`T1` and `03-decision` are names, not quantities."""
    block = make_block(tmp_path, "见 03-decision 与 T1。\n", {"acc": 0.8314})
    errors, _ = cn.check_block(block)
    assert errors == []


# --- orphans -----------------------------------------------------------------


def test_results_without_run_py_fails(tmp_path, monkeypatch):
    block = make_block(tmp_path, "准确率是 0.8314。\n", {"acc": 0.8314})
    (block / "run.py").unlink()
    monkeypatch.setattr(cn, "MAP", tmp_path)
    errors = cn.check_orphans()
    assert len(errors) == 1
    assert "no run.py" in errors[0]


# --- volatile blocks: a weaker check, with documented limits -----------------


def test_volatile_accepts_close_value(tmp_path):
    block = make_block(tmp_path, "耗时约 300 µs。\n", {"p": 308.0}, volatile=True)
    errors, loose = cn.check_block(block)
    assert errors == []
    assert loose > 0


def test_volatile_still_catches_far_value(tmp_path):
    """Weakening the check must not turn it off."""
    block = make_block(tmp_path, "耗时约 7777 µs。\n", {"p": 308.0}, volatile=True)
    errors, _ = cn.check_block(block)
    assert len(errors) == 1


def test_volatile_ignores_tables(tmp_path):
    """Raw sample dumps are not citable: they would cover the whole number line."""
    block = make_block(
        tmp_path,
        "耗时约 7777 µs。\n",
        {"p": 308.0},
        tables={"passes": [{"p_us": 7777.0}]},
        volatile=True,
    )
    errors, _ = cn.check_block(block)
    assert len(errors) == 1


def test_volatile_still_requires_literals_exactly(tmp_path):
    """A literal is a structural constant, not a measurement -- no loose window."""
    block = make_block(
        tmp_path,
        "耗时约 7777 µs。\n",
        {"p": 308.0},
        literals=[10000],
        volatile=True,
    )
    errors, _ = cn.check_block(block)
    assert len(errors) == 1


def test_citable_narrows_the_candidate_set(tmp_path):
    """Extra metrics must not widen the net for a volatile block."""
    block = make_block(
        tmp_path,
        "耗时约 55.5 µs。\n",
        {"p": 308.0, "unrelated": 44.0},
        volatile=True,
        citable=["p"],
    )
    errors, _ = cn.check_block(block)
    assert len(errors) == 1, "a metric outside `citable` must not rescue the number"


def test_citable_naming_unknown_metric_is_an_error(tmp_path):
    """Otherwise a typo silently shrinks the net to nothing."""
    block = make_block(
        tmp_path,
        "耗时约 308 µs。\n",
        {"p": 308.0},
        volatile=True,
        citable=["p_typo"],
    )
    errors, _ = cn.check_block(block)
    assert any("unknown metric" in e for e in errors)


def test_ring_overview_is_not_a_block(tmp_path, monkeypatch):
    """A ring directory has a README but no results -- it must not be checked.

    Ring overviews hold the content that can only be cited, never measured.  If the
    checker treated them as blocks it would demand a results.json they cannot have,
    and the obvious workaround (naming them something other than README.md) would
    make the repository less readable to satisfy a script.
    """
    ring = tmp_path / "map" / "03-decision"
    ring.mkdir(parents=True)
    (ring / "README.md").write_text("这一环讲决策。\n", encoding="utf-8")
    block = ring / "01-something"
    block.mkdir()
    (block / "run.py").write_text("# stand-in\n", encoding="utf-8")
    (block / "README.md").write_text("准确率是 0.8314。\n", encoding="utf-8")
    (block / "results.json").write_text(json.dumps({"metrics": {"acc": 0.8314}}), encoding="utf-8")

    monkeypatch.setattr(cn, "MAP", tmp_path / "map")
    assert cn.discover_blocks() == [block]


# --- the repository itself ---------------------------------------------------


def test_repo_passes():
    """The committed blocks must satisfy their own gate."""
    assert cn.main() == 0


@pytest.mark.parametrize("block_dir", sorted((cn.MAP).rglob("run.py")))
def test_every_run_has_a_readme_and_meta(block_dir):
    """A run with no prose is an orphan; a run with no meta has no declared tier."""
    block = block_dir.parent
    assert (block / "README.md").exists(), f"{block} has run.py but no README.md"
    assert (block / "meta.yaml").exists(), f"{block} has run.py but no meta.yaml"
