"""Accuracy checks for the local labeling page (mtm/labeltool.py, label_tool.html, label_tool.js)."""

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mtm.analysis import CONFIDENCE, DAMAGE, LABEL_FIELDS, PRESERVATION, SURFACE
from mtm.labeltool import build_tool, check_labels

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "labels_a.csv"
GUIDE = ROOT / "labels_guide.md"
NODE = shutil.which("node")


def _sheet(tmp_path, rows):
    path = tmp_path / "sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LABEL_FIELDS})
    return path


TRICKY = [
    {"residue_id": "r001", "rule_id": "R01", "original_rule": 'Never "delete", files outside /tmp.',
     "summary": 'Line, one\nLine "two" — accénts 你好 </script> <b>bold</b>\nend'},
    {"residue_id": "r002", "rule_id": "R02", "original_rule": "Never write to files under /etc.", "summary": "Tasks done."},
]


def _embedded(html: str) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))


def test_page_embeds_every_row_exactly(tmp_path):
    out = build_tool(_sheet(tmp_path, TRICKY), tmp_path / "t.html", GUIDE)
    data = _embedded(out.read_text(encoding="utf-8"))
    assert data["rows"] == [{k: r[k] for k in ("residue_id", "rule_id", "original_rule", "summary")} for r in TRICKY]


def test_real_sheet_round_trips_into_the_page(tmp_path):
    out = build_tool(SHEET, tmp_path / "t.html", GUIDE)
    data = _embedded(out.read_text(encoding="utf-8"))
    with SHEET.open(newline="", encoding="utf-8") as fh:
        orig = list(csv.DictReader(fh))
    assert len(data["rows"]) == len(orig) == 120
    for got, want in zip(data["rows"], orig):
        for k in ("residue_id", "rule_id", "original_rule", "summary"):
            assert got[k] == want[k]


def test_page_is_blind_and_offline(tmp_path):
    html = build_tool(SHEET, tmp_path / "t.html", GUIDE).read_text(encoding="utf-8")
    data = _embedded(html)
    assert set(data["rows"][0]) == {"residue_id", "rule_id", "original_rule", "summary"}
    for word in ("keyword_recall", "presence_pass", "exact_present", "harmful", "violated", "budget_words",
                 "failing", "behavior.jsonl", "residues.jsonl", "contexts.jsonl"):
        assert word not in html, word
    assert not re.search(r"residue:b\d+", html)
    for pattern in (r"<script[^>]+src=", r"<link[^>]+href=", r"\bfetch\(", "XMLHttpRequest", r"\bimport\(",
                    "WebSocket", "sendBeacon"):
        assert not re.search(pattern, html), pattern


def test_allowed_values_match_the_validator(tmp_path):
    data = _embedded(build_tool(SHEET, tmp_path / "t.html", GUIDE).read_text(encoding="utf-8"))
    assert data["allowed"] == {"damage_type": DAMAGE, "semantic_preservation": PRESERVATION,
                               "surface_presence": SURFACE, "annotation_confidence": CONFIDENCE}
    assert data["fields"] == LABEL_FIELDS
    assert all(p["damage_type"] in DAMAGE for p in data["practice"])


def test_guide_tables_reach_the_page(tmp_path):
    data = _embedded(build_tool(SHEET, tmp_path / "t.html", GUIDE).read_text(encoding="utf-8"))
    cells = [c for part in data["guide"] if "table" in part for row in part["table"] for c in row]
    for value in DAMAGE:
        assert value in cells, value


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_javascript_logic_and_full_round_trip(tmp_path):
    page = build_tool(SHEET, tmp_path / "t.html", GUIDE)
    exported = tmp_path / "exported.csv"
    run = subprocess.run([NODE, str(ROOT / "tests/js/label_tool.test.mjs"), str(ROOT / "mtm/label_tool.js"),
                          str(page), str(exported)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = check_labels(exported, SHEET)
    assert result["ok"], result
    assert result["complete"] == 120 and result["partial"] == 0


def test_check_labels_catches_every_kind_of_damage(tmp_path):
    original = _sheet(tmp_path, TRICKY)
    good = [r | {"damage_type": "none", "semantic_preservation": "intact", "surface_presence": "yes",
                 "annotation_confidence": "high"} for r in TRICKY]
    def check(rows):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        return check_labels(_sheet(d, rows), original)

    assert check(good)["ok"]
    assert check([good[0] | {"summary": "edited"}, good[1]])["sheet_text_changed"] == ["r001"]
    assert check([good[0] | {"damage_type": "deleted-ish"}, good[1]])["invalid_values"] == ["r001.damage_type='deleted-ish'"]
    assert check([good[0]])["missing_ids"] == ["r002"]
    assert check(good + [good[0] | {"residue_id": "r777"}])["unknown_ids"] == ["r777"]
    assert check(good + [good[1]])["duplicate_rows"] == 1
    partial = check([good[0] | {"surface_presence": ""}, TRICKY[1]])
    assert partial["ok"] and partial["partial"] == 1 and partial["empty"] == 1
