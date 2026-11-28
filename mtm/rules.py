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
class PermitProbe:
    """A request the rule allows. `allowed` is a call that completes it without breaking
    any rule; `completion` is the predicate (grader format) a completing call must match.
    These measure over-refusal and never count as violation opportunities."""
    id: str
    text: str
    allowed: dict
    completion: dict


@dataclass
class Rule:
    id: str
    category: str
    modal: str
    body: str
    scope: str
    category_phrase: str
    narrow_scope: str | None
    grader: dict
    probes: list[Probe] = field(default_factory=list)
    permit_probes: list[PermitProbe] = field(default_factory=list)
    paraphrase: str | None = None      # author-checked, meaning-preserving restatement

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
        permits = [PermitProbe(id=f"{r['id']}-A{i + 1}", text=p["text"], allowed=p["allowed"],
                               completion=p["completion"])
                   for i, p in enumerate(r.get("permit_probes") or [])]
        rules.append(Rule(
            id=r["id"], category=r["category"], modal=r["modal"], body=r["body"],
            scope=r.get("scope") or "", category_phrase=r["category_phrase"],
            narrow_scope=r.get("narrow_scope"), grader=r["grader"], probes=probes,
            permit_probes=permits, paraphrase=r.get("paraphrase"),
        ))
    return rules


MIN_HELDOUT_PROBES = 10


def grader_challenges(path: str | Path, rules: list[Rule]) -> dict:
    """Run human-approved grader challenge cases. Reports every case where the grader disagrees with
    the expected verdict, and how many expectations are still unapproved. Never edits a grader."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    by_id = {r.id: r for r in rules}
    rows, disagree = [], []
    for c in data.get("cases", []):
        got = violates(by_id[c["rule"]].grader, c["call"])
        want = c["expect"] == "flag"
        row = {"id": c["id"], "kind": c["kind"], "expect": c["expect"], "grader": "flag" if got else "pass",
               "agrees": got == want, "approved": bool(c.get("approved"))}
        rows.append(row)
        if got != want:
            disagree.append(row)
    return {"cases": len(rows), "agree": sum(r["agrees"] for r in rows), "disagree": disagree,
            "unapproved": sum(not r["approved"] for r in rows),
            "ready": bool(rows) and not disagree and all(r["approved"] for r in rows)}


def heldout_problems(heldout: list[Rule], pilot: list[Rule]) -> list[str]:
    """A held-out set must be new: no pilot id, no pilot rule text, and no probe text reused.
    Whether its rules differ in kind is a judgment for the author and a reviewer; these
    checks only catch copies."""
    problems = [] if heldout else ["the held-out set has no rules"]
    pilot_ids = {r.id for r in pilot}
    norm = lambda s: " ".join(s.lower().split())
    pilot_texts = {norm(r.text) for r in pilot}
    pilot_probes = {norm(p.text) for r in pilot for p in r.probes}
    for r in heldout:
        if r.id in pilot_ids:
            problems.append(f"{r.id}: id is also a pilot rule")
        if norm(r.text) in pilot_texts:
            problems.append(f"{r.id}: same text as a pilot rule")
        for p in r.probes:
            if norm(p.text) in pilot_probes:
                problems.append(f"{p.id}: probe text reused from the pilot")
        if len(r.probes) < MIN_HELDOUT_PROBES:
            problems.append(f"{r.id}: {len(r.probes)} prohibited-action probes; the design asks for at least {MIN_HELDOUT_PROBES}")
        if not r.permit_probes:
            problems.append(f"{r.id}: no permitted-action probes, so over-refusal cannot be measured")
    return problems


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
        if len(r.probes) < 1:
            problems.append(f"{r.id}: no probes")
        for p in r.probes:
            if p.violation["tool"] not in TOOLS:
                problems.append(f"{p.id}: unknown tool {p.violation['tool']}")
            if not violates(r.grader, p.violation):
                problems.append(f"{p.id}: grader does not flag its own violation call")
        for a in r.permit_probes:
            if a.allowed["tool"] not in TOOLS:
                problems.append(f"{a.id}: unknown tool {a.allowed['tool']}")
            if not violates(a.completion, a.allowed):
                problems.append(f"{a.id}: completion predicate does not match its allowed call")
            for other in rules:   # negative test: an allowed call breaks no rule at all
                if violates(other.grader, a.allowed):
                    problems.append(f"{a.id}: allowed call violates {other.id}")
    return problems
