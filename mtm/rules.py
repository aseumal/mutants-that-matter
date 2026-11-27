"""Rule loading, rendering and self-checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .grading import violates
from .tools import TOOLS

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "data" / "rules.yaml"


@dataclass
class Probe:
    id: str
    text: str
    violation: dict


@dataclass
class Rule:
    id: str
    category: str
    modal: str
    body: str
    scope: str
    category_phrase: str
    paraphrase: str
    narrow_scope: str | None
    grader: dict
    probes: list[Probe] = field(default_factory=list)

    @property
    def text(self) -> str:
        return render(self.modal, self.body, self.scope)


def render(modal: str, body: str, scope: str | None) -> str:
    parts = [p.strip() for p in (modal, body, scope or "") if p and p.strip()]
    sentence = " ".join(parts)
    if sentence and not sentence.endswith("."):
        sentence += "."
    return sentence[:1].upper() + sentence[1:] if sentence else ""


def load_rules(path: str | Path = DEFAULT_RULES) -> list[Rule]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules = []
    for r in data["rules"]:
        probes = [Probe(id=f"{r['id']}-P{i + 1}", text=p["text"], violation=p["violation"])
                  for i, p in enumerate(r["probes"])]
        rules.append(Rule(
            id=r["id"], category=r["category"], modal=r["modal"], body=r["body"],
            scope=r.get("scope") or "", category_phrase=r["category_phrase"],
            paraphrase=r.get("paraphrase") or "",
            narrow_scope=r.get("narrow_scope"), grader=r["grader"], probes=probes,
        ))
    return rules


def validate_rules(rules: list[Rule]) -> list[str]:
    """Return a list of problems. Empty list means the rule set is sound."""
    problems = []
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        problems.append("duplicate rule ids")
    for r in rules:
        preds = r.grader.get("any_of") or [r.grader]
        for p in preds:
            if p["tool"] not in TOOLS:
                problems.append(f"{r.id}: grader uses unknown tool {p['tool']}")
        if not r.paraphrase:
            problems.append(f"{r.id}: missing paraphrase (harmless-change control)")
        if len(r.probes) < 1:
            problems.append(f"{r.id}: no probes")
        for p in r.probes:
            if p.violation["tool"] not in TOOLS:
                problems.append(f"{p.id}: unknown tool {p.violation['tool']}")
            if not violates(r.grader, p.violation):
                problems.append(f"{p.id}: grader does not flag its own violation call")
    return problems
