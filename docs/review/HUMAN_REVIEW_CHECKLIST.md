# Human review before the feasibility run

Code checks cannot establish meaning. These checks must be done by a person before paid replay. Ideally
that is someone other than the tool author and other than the residue annotator. If you do them yourself,
do them **after** finishing the 120 residue labels, because the span sheet shows how the extractor read
some of those summaries.

## 1. Policy spans of the 12 feasibility residues
- **File:** `docs/review/feasibility_spans_REVIEW.csv`. Items are shuffled and show no replay outcome. `feasibility_spans_KEY.json` maps item to residue.
- **Check each item for:**
  - Is the extracted span exactly the text that carries this rule, no more and no less?
  - Is any policy language left in the context?
  - Is the sham sentence free of any policy?
- **Where a span is wrong,** write the correct one to `runs/stage2-feasibility-v3/policy_spans.jsonl` after the import, one line per residue:
  `{"rule_id": "R14", "condition": "residue:b60:c2", "spans": ["<exact text>"]}`
  Manual spans override the heuristic and are marked `source: manual`.
- **Pass:** every item answered, and every correction written.

## 2. Paraphrases (condition C5, optional)
- **File:** `docs/review/paraphrases_DRAFT.yaml`. Drafted by Claude and not checked.
- **For each paraphrase:** does it express exactly the same policy, with the same action, scope and exceptions, and no softer force?
- Approve, edit or reject each one.
- C5 stays off until at least one paraphrase is approved and copied into `data/rules.yaml`.

## 3. Graders of the 12 rules
For each rule, read its grader in `data/rules.yaml` and try to break it:
- Write at least 3 calls that **should** be flagged, including one through `run_shell`.
- Write at least 3 calls that **should not** be flagged, including one boundary case.
- Run each through `python -c "from mtm.grading import violates; ..."`.

Record any disagreement. A grader change needs a new run name and a regrade check (`mtm import-run` refuses a source whose verdicts change).

## 4. Raw traces (after labeling)
Read 30 v3 episodes from the 12 rules, stratified by condition type, and confirm that each stored verdict matches what the agent actually did. This shows replay outcomes, so do it after the 120 labels are done.

## 5. Spending cap
Agree a spending cap in writing before running. The token estimate is in `PREREGISTRATION_DRAFT.md`. From the first real run on, measured usage is appended to `runs/<run>/usage.jsonl`.
