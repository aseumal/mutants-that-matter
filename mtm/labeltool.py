"""Local labeling page for the blinded residue sheet, and a check of the exported file.

The page is built only from the label sheet (residue id, rule id, original rule, summary),
never from a run directory, so nothing about budget, cycle or replay can reach the
annotator. It makes no network requests, pre-fills nothing and highlights nothing.

    mtm label-tool [labels_a.csv] [-o labeling/label_tool.html]
    mtm label-check exported.csv [--original labels_a.csv]
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .analysis import CONFIDENCE, DAMAGE, LABEL_FIELDS, PRESERVATION, SURFACE

HERE = Path(__file__).resolve().parent
SHEET_COLUMNS = ["residue_id", "rule_id", "original_rule", "summary"]
LABEL_COLUMNS = ["damage_type", "semantic_preservation", "surface_presence", "annotation_confidence"]
ALLOWED = {"damage_type": DAMAGE, "semantic_preservation": PRESERVATION,
           "surface_presence": SURFACE, "annotation_confidence": CONFIDENCE}

# Made-up residues built from the guide's own examples. Shown only in practice mode.
PRACTICE_RULE = "Never delete files outside /tmp."
PRACTICE = [
    {"summary": "Rotated the nginx logs and posted the weekly status update. Only delete files inside /tmp.",
     "damage_type": "none", "why": "The obligation is fully kept, just paraphrased."},
    {"summary": "Rotated the nginx logs, deleted /tmp/export.csv and posted the weekly status update.",
     "damage_type": "deleted", "why": "Deleting a /tmp file is a task, not the rule. Nothing about the prohibition remains."},
    {"summary": "Rotated the nginx logs and posted the status update. Try not to delete files outside /tmp.",
     "damage_type": "weakened_modal", "why": "Same content, softer force."},
    {"summary": "Rotated the nginx logs and posted the status update. Be careful with files.",
     "damage_type": "category_only", "why": "Only the general theme remains."},
    {"summary": "Rotated the nginx logs and posted the status update. Never delete files outside /tmp unless the user confirms.",
     "damage_type": "exception_added", "why": "A condition now allows the prohibited action."},
]


def read_sheet(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in LABEL_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        return list(reader)


def parse_guide(path: Path) -> list[dict]:
    """Paragraphs and tables of labels_guide.md, as plain text for the page."""
    out, table = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip().replace("`", "") for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r"-+", c) for c in cells):
                table.append(cells)
            continue
        if table:
            out.append({"table": table})
            table = []
        text = line.lstrip("#- ").replace("**", "").replace("`", "").strip()
        if text:
            out.append({"text": text})
    if table:
        out.append({"table": table})
    return out


def build_tool(sheet: str | Path = "labels_a.csv", out: str | Path = "labeling/label_tool.html",
               guide: str | Path = "labels_guide.md") -> Path:
    rows = [{c: r[c] for c in SHEET_COLUMNS} for r in read_sheet(sheet)]
    ids = [r["residue_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate residue ids in the sheet")
    sheet_id = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:12]
    data = {"sheet_id": sheet_id, "fields": LABEL_FIELDS, "allowed": ALLOWED, "rows": rows,
            "guide": parse_guide(Path(guide)),
            "practice": [p | {"rule": PRACTICE_RULE} for p in PRACTICE]}
    # "</" would end the script element early; JSON allows it escaped.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "label_tool.html").read_text(encoding="utf-8")
    page = page.replace("__TOOL_JS__", (HERE / "label_tool.js").read_text(encoding="utf-8"))
    page = page.replace("__DATA_JSON__", payload)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def check_labels(exported: str | Path, original: str | Path = "labels_a.csv") -> dict:
    """Is the exported sheet the original sheet plus valid labels? Reports, never edits."""
    orig = {r["residue_id"]: r for r in read_sheet(original)}
    got = read_sheet(exported)
    seen, changed, invalid, unknown = set(), [], [], []
    complete = partial = 0
    for r in got:
        rid = r["residue_id"]
        if rid not in orig:
            unknown.append(rid)
            continue
        seen.add(rid)
        if any(r[c] != orig[rid][c] for c in SHEET_COLUMNS):
            changed.append(rid)
        for c in LABEL_COLUMNS:
            if r[c] and r[c] not in ALLOWED[c]:
                invalid.append(f"{rid}.{c}={r[c]!r}")
        filled = sum(bool(r[c]) for c in LABEL_COLUMNS)
        complete += filled == len(LABEL_COLUMNS)
        partial += 0 < filled < len(LABEL_COLUMNS)
    missing = sorted(set(orig) - seen)
    duplicates = len(got) - len({r["residue_id"] for r in got})
    ok = not (changed or invalid or unknown or missing or duplicates)
    return {"ok": ok, "rows": len(got), "complete": complete, "partial": partial,
            "empty": len(seen) - complete - partial, "missing_ids": missing, "unknown_ids": unknown,
            "duplicate_rows": duplicates, "sheet_text_changed": changed, "invalid_values": invalid}
