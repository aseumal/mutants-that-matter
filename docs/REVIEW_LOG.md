# Review log

Concerns raised by external reviews of the manuscript and how each version answered them. Moved out of the paper on 2026-11-28 at a reviewer's suggestion.

| Concern | Response |
|---|---|
| High coupling could be chance | Chance and difficulty-aware baselines (Lemma 1, Table) |
| Mutation score is confounded with suite size | Suites at fixed size per rule, with a deletion-only score (Table) |
| Deletion may make mutants unnecessary | Deletion is the primary comparator throughout |
| Equal probes are not equal cost | Candidate episodes and cost per extra residue (Table) |
| The two-of-three threshold is fragile | Lemma 2 and a full threshold sweep (Table); violation probability primary in Stage 2 |
| Semantic and behavioral equivalence conflated | Definition 4; deletion-like stated as observational |
| Synthetic-to-real transfer untested | Deletion, mutant, weighted and size-matched training (Table) |
| Mutants and residues replayed in different contexts | Stated as a limitation; matched-context replay implemented for Stage 2 |
| No permitted actions; over-refusal unmeasured | Permitted-action probes implemented for Stage 2 |
| Discovery could select favorable settings | Representative and enriched samples kept apart; gate on fault supply only |
| Grader reliability | Audit with four corrections; regrading reproduces every verdict |
| Zero-failure bounds stated loosely | Lemma 3; one- and two-sided bounds both reported |
| No figure carried the main result | Figure, detection against cost |
| Pooled numbers can hide one bad set | Per-category table (Table); per-category reporting planned |
| Verdicts on point estimates | Four-way verdict with an inconclusive outcome (Appendix) |
| No positive control | Planted decay mutant in real context; pinned rule as null |
| Measuring at the wrong place | Splits checked to add back to the summary word for word |
| Positive-control logic was wrong (2026-11-28 review) | Planted mutants now read as context dependence; a mismatch with the neutral frame is a finding, not a probe failure |
| Lossless split is not correct extraction | Added leftover-constraint and cross-rule checks; sentences go to the best-matching rule; human review of spans required before matched results |
| Classifier may read a missing-words shortcut | Shortcut diagnostics: deletion-trained model does not differ from keyword recall; reported in the paper |
| Primary estimand undefined | Empirical detection estimand, equal total cost, four-way verdict; PREREGISTRATION_DRAFT.md |
| Held-out probes ambiguous | Development, held-out candidate and independent reference probe partitions |
| Zero-width intervals misleading | Reported as 0 observed, bootstrap degenerate |
| Ten samples called sufficient; unequal allocation | Balanced allocation; ten samples described as improved, not sufficient |
| Logits do not give violation probability | Repeated sampling primary; constrained-action diagnostic only |
