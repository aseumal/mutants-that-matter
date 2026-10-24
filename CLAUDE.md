# CLAUDE.md

Context for Claude Code sessions on this repo.

## Project

"Mutants That Matter: Validating Mutation Testing for Silent Guardrail Decay in LLM Agents."
Research by Anthony Seumal (PhD Data Science, AIM). Pilot due for a deep learning class
presentation in November 2026; target is a Q1 journal (ACM TOSEM or Empirical Software
Engineering). Also submitted as a TARA Round 2 project idea.

Core question: are synthetic rule mutants a valid substitute for the real damage that
context compaction (self-summarization) does to an agent's safety rules? This adapts
Just et al. (FSE 2014) and Wang et al. (TOSEM, arXiv:2406.09843) from code bugs to
agent guardrails. Builds on Chen 2026 "Governance Decay" (arXiv:2606.22528) and
Kwartler, Aqrawi & Abbasi 2026 "AI Guardrail Survival" (arXiv:2608.11392).
Closest prior tooling: muteval (prompt mutation for eval suites), whose operators are
heuristic and not validated against real compaction damage. That gap is our novelty.

## Research questions

- RQ1 Realism: how does compaction damage rules? (residues + human labels + kappa)
- RQ2 Coupling: do rule-based and LLM mutants behave like real residues? (coupling, Ochiai)
- RQ3 Audit adequacy: presence check vs LLM judge vs behavioral replay
- RQ4 Prediction: classifier on (original rule, compacted rule) pairs predicting functional survival

## Layout

- `data/rules.yaml` 30 rules (modal/body/scope parts), 3 probes each, deterministic graders
- `data/filler.yaml` rule-clean benign tasks for session history
- `mtm/` package: llm (OpenAI-compatible client + MockLLM), agent (JSON action protocol),
  session, compaction, mutants (8 operators + LLM mutants), audits (presence check),
  grading, experiment (stages, resumable JSONL), analysis (coupling, report, kappa), cli
- `configs/smoke.yaml` (5 rules, 1 sample), `configs/pilot.yaml` (30 rules, 2 budgets, 2 cycles, 3 samples)
- outputs go to `runs/<run_name>/` (gitignored)

## Commands

```bash
pip install -e ".[dev]"
mtm validate
pytest -q
mtm pilot --mock                     # plumbing check, numbers are meaningless
mtm pilot -c configs/smoke.yaml      # first real run, needs .env
mtm pilot -c configs/pilot.yaml
mtm export-labels -c configs/pilot.yaml -o labels_a.csv
mtm kappa labels_a.csv labels_b.csv
```

## Status (as of 2026-10-24)

Done: harness, 30 rules/90 probes, mutants, presence audit, coupling analysis, labeling, tests (13 passing).

Next, in order:
1. Run `configs/smoke.yaml` against a real API (endpoint not yet chosen); check JSON format
   compliance (`parse_errors` in behavior.jsonl) and cost.
2. Add the spliced-mutant control: replay each mutant spliced into its rule's real summary,
   so mutants and residues share context (README "Known limitations").
3. Add an LLM-judge audit for RQ3 (judge says intact/degraded/dropped per residue and mutant).
4. Full pilot on 2 open-weight models; hand-label residues; compute kappa.
5. RQ4 prototype: small classifier (e.g. fine-tuned sentence encoder or logistic regression on
   embeddings) on (original, compacted) pairs labeled by replay outcome. This is the deep
   learning deliverable for the class.

## Conventions

- Graders must stay deterministic. Never use an LLM to decide whether a violation happened.
- Every new rule needs probes its grader flags; `mtm validate` and tests enforce this.
- Filler tasks must never violate any rule.
- Tools are simulated only; nothing is ever executed.
- Keep stages resumable (append JSONL, skip finished rows).
