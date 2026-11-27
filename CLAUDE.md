# CLAUDE.md

Context for Claude Code sessions on this repo.

## Project

"Mutants That Matter: Validating Mutation Testing for Silent Guardrail Decay in LLM Agents."
Research by Anthony Seumal (PhD Data Science, AIM). Pilot due for a deep learning class
presentation in November 2026; target is a Q1 journal (ACM TOSEM or Empirical Software
Engineering). This is also his TARA Round 2 project.

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
  session, compaction, mutants (8 operators + paraphrase control + LLM mutants), audits (presence check),
  grading, experiment (stages, resumable JSONL), analysis (coupling, controls, calibration,
  sensitivity, report, kappa), freeze (design hashing), cli
- `PROTOCOL.md` pre-registration: what is frozen, controls, thresholds, independence
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
mtm freeze --note "..."            # before any real run
mtm check-freeze
```

## Status (as of 2026-11-27)

Done: harness, 30 rules/90 probes, mutants, presence audit, coupling analysis, labeling.
Added 2026-11-27 (lessons from his drought class paper, "Green Metrics, Broken Pipelines"):
paraphrase control operator, intact_repeat noise condition, mutant direction classes,
per-probe and grid CSVs, tau calibration and sensitivity, `mtm freeze` / `check-freeze`,
PROTOCOL.md. Tests: 20 passing.

Next, in order:
1. Review PROTOCOL.md, then `mtm freeze` and commit FROZEN.json BEFORE any real run.
2. Run `configs/smoke.yaml` against a real API (endpoint not yet chosen); check JSON format
   compliance (`parse_errors` in behavior.jsonl) and cost. Use the pilot's calibrated tau
   for the full study.
3. Add the spliced-mutant control: replay each mutant spliced into its rule's real summary,
   so mutants and residues share context (README "Known limitations").
4. Add an LLM-judge audit for RQ3 (judge says intact/degraded/dropped per residue and mutant).
5. Full pilot on 2 open-weight models; hand-label residues; compute kappa.
6. RQ4 prototype: small classifier (e.g. fine-tuned sentence encoder or logistic regression on
   embeddings) on (original, compacted) pairs labeled by replay outcome. This is the deep
   learning deliverable for the class.

## Conventions

- Graders must stay deterministic. Never use an LLM to decide whether a violation happened.
- Every new rule needs probes its grader flags; `mtm validate` and tests enforce this.
- Filler tasks must never violate any rule.
- Tools are simulated only; nothing is ever executed.
- Keep stages resumable (append JSONL, skip finished rows).
- Controls (paraphrase, drop_scope, intact_repeat) are false-alarm measures, never coupling evidence.
- After `mtm freeze`, changes to frozen files are a new protocol version; anything added after
  seeing residues is reported as adaptive, not blind.
- Report per probe as well as averaged; never only pooled numbers.
