"""Matched-context replay: separate a compaction summary into the text that carries a
rule (the policy span) and the surrounding context, so that the intact rule, the
residue, a deletion and each mutant can be replayed inside the same real context.

Span detection is a deterministic heuristic, not ground truth. A damaged rule can be
spread across a summary or survive only as an allusion, so every split records what
it removed and two warning flags:

- no_span            nothing rule-like was found, so the context is the whole summary
- residual_reference the context still shares content words with the rule after removal
- embedded_span      a span longer than EMBEDDED_WORDS words: the rule likely survives only as an
                     aside inside a task list ("not admin, per operator policy"), so removing the
                     whole segment also removes ordinary context. Check these by hand.

Hand-checked spans override the heuristic: put rows of
{"rule_id", "condition", "spans": [...]} in <run>/policy_spans.jsonl. Analyses should
report results with and without flagged splits.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rules import Rule

# Segments end at sentence punctuation, semicolons or line breaks. Separators are kept
# so the context reads as the compactor wrote it.
_SEGMENT = re.compile(r"(?<=[.!?;])\s+|(?<=[.!?;][\"'\u201d)\]])\s+|\n+")
EMBEDDED_WORDS = 30
_POLICY_CUE = re.compile(
    r"\b(never|not|no|n't|must|always|only|avoid|polic(y|ies)|rules?|constraints?|guidelines?|"
    r"prohibit\w*|forbid\w*|restrict\w*|requir\w*|approv\w*|careful|caution|unless|without)\b", re.I)
_STOP = {"the", "a", "an", "to", "of", "or", "and", "any", "in", "on", "for", "with", "that",
         "than", "from", "by", "never", "not", "do", "must", "be", "is", "are", "when", "other",
         "into", "this", "all", "only"}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9#/._@+-]+", text.lower())
    return {w[:5] if len(w) > 5 else w for w in words if len(w) > 2 and w not in _STOP}


def rule_stems(rule: Rule) -> set[str]:
    return _stems(f"{rule.body} {rule.scope}")


def segments(text: str) -> list[tuple[str, str]]:
    """[(segment, separator_after)] covering the whole text."""
    out, pos = [], 0
    for m in _SEGMENT.finditer(text):
        out.append((text[pos:m.start()], m.group(0)))
        pos = m.end()
    out.append((text[pos:], ""))
    return [(s, sep) for s, sep in out if s.strip() or sep]


def is_policy_span(rule: Rule, segment: str) -> bool:
    overlap = len(rule_stems(rule) & _stems(segment))
    return overlap >= 2 or (overlap >= 1 and _POLICY_CUE.search(segment) is not None)


@dataclass
class Split:
    rule_id: str
    condition: str
    context: str
    spans: list[str] = field(default_factory=list)
    no_span: bool = False
    residual_reference: bool = False
    source: str = "heuristic"
    embedded_span: bool = False
    sham_removed: str | None = None   # a non-policy segment removed for the sham-edit control
    sham_summary: str | None = None   # the real summary minus that segment, policy span kept

    @property
    def residue_text(self) -> str:
        return " ".join(s.strip() for s in self.spans)


def split_summary(rule: Rule, condition: str, summary: str,
                  manual_spans: list[str] | None = None) -> Split:
    if manual_spans is not None:
        context = summary
        for span in manual_spans:
            context = context.replace(span, " ")
        spans, source = list(manual_spans), "manual"
    else:
        kept, spans = [], []
        for seg, sep in segments(summary):
            if seg.strip() and is_policy_span(rule, seg):
                spans.append(seg.strip())
            else:
                kept.append(seg + sep)
        context, source = "".join(kept), "heuristic"
    context = re.sub(r"[ \t]+", " ", context).strip()
    residual = bool(rule_stems(rule) & _stems(context)) and _POLICY_CUE.search(context) is not None
    removed, sham = sham_edit(rule, summary, spans)
    embedded = source == "heuristic" and any(len(s.split()) > EMBEDDED_WORDS for s in spans)
    return Split(rule.id, condition, context, spans, no_span=not spans, residual_reference=residual,
                 source=source, embedded_span=embedded, sham_removed=removed, sham_summary=sham)


def sham_edit(rule: Rule, summary: str, spans: list[str]) -> tuple[str | None, str | None]:
    """Remove one segment that carries no policy, about as long as the policy span (or of
    median length when there is none), and keep everything else. Replaying this tells
    whether editing a summary changes behavior apart from what the edit says."""
    segs = segments(summary)
    candidates = [i for i, (s, _) in enumerate(segs)
                  if len(s.split()) >= 3 and s.strip() not in spans and not is_policy_span(rule, s)]
    if not candidates:
        return None, None
    lengths = sorted(len(segs[i][0].split()) for i in candidates)
    target = len(" ".join(spans).split()) if spans else lengths[len(lengths) // 2]
    pick = min(candidates, key=lambda i: (abs(len(segs[i][0].split()) - target), i))
    text = "".join(s + sep for k, (s, sep) in enumerate(segs) if k != pick)
    return segs[pick][0].strip(), re.sub(r"[ \t]+", " ", text).strip()


def load_manual_spans(run_dir: Path) -> dict[tuple[str, str], list[str]]:
    path = Path(run_dir) / "policy_spans.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["rule_id"], r["condition"]): r["spans"] for r in rows}


def build_splits(rules: list[Rule], residues: list[dict], run_dir: Path, write: bool = True) -> list[dict]:
    """One split per residue, written to <run>/contexts.jsonl for auditing."""
    by_id = {r.id: r for r in rules}
    manual = load_manual_spans(run_dir)
    rows = []
    for res in residues:
        rule = by_id.get(res["rule_id"])
        if rule is None:
            continue
        key = (res["rule_id"], res["condition"])
        rows.append(asdict(split_summary(rule, res["condition"], res["summary"], manual.get(key))))
    if not write:
        return rows
    path = Path(run_dir) / "contexts.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return rows
