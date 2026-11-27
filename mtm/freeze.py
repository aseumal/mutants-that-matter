"""Freeze the study design before looking at real residues.

The drought study's reviewers marked guardrails written after seeing the faults
as "adaptive, not blind". To keep this study blind, the rules, probes, graders,
mutation operators, audits and configs are hashed into FROZEN.json *before* the
first real compaction run. Every run records whether it matched the freeze.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "FROZEN.json"

# What must not change once data collection starts.
FROZEN_PATHS = [
    "data/rules.yaml",
    "data/filler.yaml",
    "mtm/grading.py",
    "mtm/mutants.py",
    "mtm/audits.py",
    "mtm/compaction.py",
    "mtm/agent.py",
    "configs/pilot.yaml",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def current_hashes() -> dict[str, str]:
    return {p: _sha(ROOT / p) for p in FROZEN_PATHS if (ROOT / p).exists()}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def freeze(note: str = "") -> dict:
    record = {
        "frozen_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "note": note,
        "hashes": current_hashes(),
    }
    FROZEN.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check() -> dict:
    """{'frozen': bool, 'matches': bool | None, 'changed': [paths]}"""
    if not FROZEN.exists():
        return {"frozen": False, "matches": None, "changed": []}
    record = json.loads(FROZEN.read_text(encoding="utf-8"))
    now = current_hashes()
    changed = sorted(p for p in set(record["hashes"]) | set(now)
                     if record["hashes"].get(p) != now.get(p))
    return {"frozen": True, "matches": not changed, "changed": changed,
            "frozen_at": record.get("frozen_at"), "git_commit": record.get("git_commit")}
