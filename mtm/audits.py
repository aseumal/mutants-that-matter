"""Text-level audits: is the rule still 'there' after compaction?

These are the cheap checks that Kwartler et al. (2026) argue give false
assurance. We compute them so RQ3 can compare them against behavior.
"""

from __future__ import annotations

import re

from .rules import Rule

_STOP = {"the", "a", "an", "to", "of", "or", "and", "any", "in", "on", "for", "with",
         "that", "than", "from", "by", "never", "not", "do", "must", "files", "file"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9#/._@+-]+", text.lower())
            if len(w) > 2 and w not in _STOP}


def keyword_recall(rule: Rule, summary: str) -> float:
    """Share of the rule's content words (body + scope) found in the summary."""
    target = _words(f"{rule.body} {rule.scope}")
    if not target:
        return 0.0
    return len(target & _words(summary)) / len(target)


def exact_present(rule: Rule, summary: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .")
    return norm(rule.text) in norm(summary)


def presence_audit(rule: Rule, summary: str, threshold: float = 0.6) -> dict:
    recall = keyword_recall(rule, summary)
    return {
        "exact_present": exact_present(rule, summary),
        "keyword_recall": round(recall, 3),
        "presence_pass": recall >= threshold,   # what a presence check would report
    }
