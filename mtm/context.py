"""Matched-context replay: separate a compaction summary into the text that carries a
rule (the policy span) and the surrounding context, so that the intact rule, the
residue, a deletion and each mutant can be replayed inside the same real context.

Span detection is a deterministic heuristic, not ground truth. A damaged rule can be
spread across a summary or survive only as an allusion, so every split records what
it removed and two warning flags:

- no_span            nothing rule-like was found, so the context is the whole summary
- residual_reference the context still shares content words with the rule after removal
- context_policy     the context still contains constraint language ("must", "never", "policy"...)
                     that no other rule of the same session explains: possibly a missed piece of
                     this rule. Other session rules' text is expected and counted separately. A lossless split
                     guarantees no text was dropped; it does not guarantee the right text was moved.
- ambiguous_spans    sentences that match this rule and another session rule equally well. They
                     stay in the context, and the split is excluded from matched replay until a
                     human assigns them in policy_spans.jsonl
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
_CONSTRAINT = re.compile(r"\b(never|must|do not|don't|not allowed|prohibit\w*|forbid\w*|polic(y|ies)|rules?|"
                         r"restrict\w*|requires? approval|only (?:if|when|after|with))\b", re.I)
_POLICY_CUE = re.compile(
    r"\b(never|not|no|n't|must|always|only|avoid|polic(y|ies)|rules?|constraints?|guidelines?|"
    r"prohibit\w*|forbid\w*|restrict\w*|requir\w*|approv\w*|careful|caution|unless|without)\b", re.I)
_STOP = {"the", "a", "an", "to", "of", "or", "and", "any", "in", "on", "for", "with", "that",
         "than", "from", "by", "never", "not", "do", "must", "be", "is", "are", "when", "other",
         "into", "this", "all", "only"}


def _stems(text: str) -> set[str]:
    # trailing sentence punctuation is not part of a word: "/etc." must match "/etc"
    words = [w.rstrip(".") for w in re.findall(r"[a-z0-9#/._@+-]+", text.lower())]
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
    context_policy: bool = False
    other_rule_spans: int = 0
    ambiguous_spans: int = 0
    sham_removed: str | None = None   # a non-policy segment removed for the sham-edit control
    sham_summary: str | None = None   # the real summary minus that segment, policy span kept

    @property
    def residue_text(self) -> str:
        return " ".join(s.strip() for s in self.spans)


def split_summary(rule: Rule, condition: str, summary: str,
                  manual_spans: list[str] | None = None, other_rules: list[Rule] | None = None) -> Split:
    ambiguous = 0
    if manual_spans is not None:
        context = summary
        for span in manual_spans:
            context = context.replace(span, " ")
        spans, source = list(manual_spans), "manual"
    else:
        kept, spans, ambiguous = [], [], 0
        own = rule_stems(rule)
        for seg, sep in segments(summary):
            if seg.strip() and is_policy_span(rule, seg):
                # in a shared session a sentence belongs to the rule it matches best, so editing
                # one rule never removes another rule's text. A tie stays in the context and is
                # flagged: it needs a human decision (policy_spans.jsonl) before any edit.
                mine = len(own & _stems(seg))
                best_other = max((len(rule_stems(o) & _stems(seg)) for o in other_rules or []
                                  if is_policy_span(o, seg)), default=-1)
                if best_other >= mine:
                    ambiguous += best_other == mine
                    kept.append(seg + sep)
                    continue
                spans.append(seg.strip())
            else:
                kept.append(seg + sep)
        context, source = "".join(kept), "heuristic"
    context = re.sub(r"[ \t]+", " ", context).strip()
    residual = bool(rule_stems(rule) & _stems(context)) and _POLICY_CUE.search(context) is not None
    removed, sham = sham_edit(rule, summary, spans)
    embedded = source == "heuristic" and any(len(s.split()) > EMBEDDED_WORDS for s in spans)
    others = [seg for seg, _ in segments(context)
              if seg.strip() and any(is_policy_span(o, seg) for o in other_rules or [])]
    unexplained = [seg for seg, _ in segments(context)
                   if seg.strip() and _CONSTRAINT.search(seg) and seg not in others]
    constraint = bool(unexplained)
    return Split(rule.id, condition, context, spans, no_span=not spans, residual_reference=residual,
                 source=source, embedded_span=embedded, context_policy=constraint,
                 other_rule_spans=len(others), ambiguous_spans=ambiguous,
                 sham_removed=removed, sham_summary=sham)


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


def _words(text: str) -> list[str]:
    return sorted(re.findall(r"\S+", text))


def split_integrity(summary: str, split: dict) -> list[str]:
    """Problems with a split, empty when sound. The context plus the removed spans must hold
    exactly the summary's words, and the sham summary plus its removed segment likewise, so a
    matched condition differs from the real summary only in the edited span. Manual spans may
    reword, so only heuristic splits are checked word for word."""
    problems = []
    if split.get("source", "heuristic") == "heuristic":
        if _words(split["context"] + " " + " ".join(split["spans"])) != _words(summary):
            problems.append("context and spans do not add up to the summary")
    if split.get("sham_summary"):
        if _words(split["sham_summary"] + " " + split["sham_removed"]) != _words(summary):
            problems.append("sham summary and removed segment do not add up to the summary")
        if any(s not in split["sham_summary"] for s in split["spans"]):
            problems.append("sham edit removed part of the policy span")
    return problems


def plant(summary: str, spans: list[str], text: str) -> str | None:
    """The residue's summary with its policy span replaced, in place, by a mutant. Other spans are
    removed. None when there is no span to replace. Compared with the same mutant in the neutral
    frame, this measures how much real context changes a mutant's behavior."""
    if not spans or spans[0] not in summary:
        return None
    out = summary.replace(spans[0], text, 1)
    for s in spans[1:]:
        out = out.replace(s, " ", 1)
    return re.sub(r"[ \t]+", " ", out).strip()


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
        others = [by_id[o] for o in res.get("session_rules", []) if o != rule.id and o in by_id]
        rows.append(asdict(split_summary(rule, res["condition"], res["summary"], manual.get(key), others)))
    if not write:
        return rows
    path = Path(run_dir) / "contexts.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return rows
