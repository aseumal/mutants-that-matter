"""RQ1 tables are written only when every residue is labelled (protects blinding)."""

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "runs/pilot-haiku-v3"


def _module(tmp_path, labels):
    spec = importlib.util.spec_from_file_location("make_tables", ROOT / "paper/make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RUN, mod.OUT, mod.LABELS_CSV = V3, tmp_path, labels
    return mod


def _sheet(tmp_path, fill: int):
    rows = list(csv.DictReader((ROOT / "labels_a.csv").open(newline="", encoding="utf-8")))
    for k, r in enumerate(rows[:fill]):
        r.update(damage_type="deleted", semantic_preservation="absent", surface_presence="no",
                 annotation_confidence="high")
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.mark.skipif(not (V3 / "residues.jsonl").exists(), reason="pilot run data not present")
def test_partial_labels_write_nothing(tmp_path):
    (tmp_path / "rq1.tex").write_text("stale")
    _module(tmp_path, _sheet(tmp_path, 119)).rq1_tables()
    assert not (tmp_path / "rq1.tex").exists()


@pytest.mark.skipif(not (V3 / "residues.jsonl").exists(), reason="pilot run data not present")
def test_complete_labels_write_the_table(tmp_path):
    _module(tmp_path, _sheet(tmp_path, 120)).rq1_tables()
    text = (tmp_path / "rq1.tex").read_text()
    assert "tab:rq1" in text and "of all 120 residues" in text and "\\texttt{deleted} & 120" in text
