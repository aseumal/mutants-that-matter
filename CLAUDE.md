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
pip install -e ".[dev,ml]"            # ml is only needed for mtm predict
mtm validate
pytest -q
mtm pilot --mock                     # plumbing check, numbers are meaningless
mtm pilot -c configs/smoke.yaml      # first real run, needs .env
mtm pilot -c configs/pilot.yaml
mtm export-labels -c configs/pilot.yaml -o labels_a.csv
mtm kappa labels_a.csv labels_b.csv
mtm label-summary labels_a.csv -c configs/pilot.yaml
mtm predict -c configs/pilot.yaml    # RQ4
mtm plan -c configs/stage2_matched.yaml     # calls and episodes a config implies, no API use
mtm import-run -c configs/stage2_matched.yaml --source pilot-haiku-v3   # copy v3 only if graders reproduce it
mtm stage2 -c configs/stage2_matched.yaml   # outcomes, over-refusal, compression, matched context
mtm label-tool                              # local blinded labeling page from labels_a.csv
mtm label-check exported.csv                # verify an exported label sheet
mtm validate --rules data/heldout_rules.yaml # held-out rule set: no pilot copies, >= 10 probes, permitted probes
```

v3 is FROZEN (2026-11-28): data files are read-only, archived with SHA-256 in `archive/`; new analyses write to
`paper/analysis/results/`, never into `runs/pilot-haiku-v3`. Confirmatory protocol draft: `PREREGISTRATION_DRAFT.md`.
Feasibility config (12 residues, not run): `configs/stage2_feasibility.yaml`.
Stage 2 code (2026-11-27, not yet run): see `STUDY_DESIGN.md`. Fingerprints in `run.json` block resuming
under changed rules, probes, graders, filler, prompts or generative settings. Core analyses read only
pilot-design rows (`analysis.is_core`); Stage 2 rows go to `mtm/stage2.py`. Permitted probes were drafted by
Claude and still need the author's review.

## Status (as of 2026-11-07)

Class presentation is due by about Nov 14. Scope is this repo only, on one hosted model (Claude Haiku 5.5, not open-weight); the weight-level
safety mutants study is a separate follow-up paper.

Done: harness, 30 rules/90 probes, mutants, presence audit, coupling with chance, rule-absent and noise-floor baselines,
RQ4 prototype (`mtm/predict.py`), labeling tools. Real runs use Claude Haiku 5.5 at low effort for compactor and agent (no Groq account; budgets 30/60, since Opus kept
nearly every rule). Claude is not open-weight and rejects `temperature`, so list both as limitations.
`runs/pilot-haiku-v3` is the current run: v2 rows regraded with shell-aware graders (R01, R02, R10, R16, R17, R28) and
capped episodes continued to 8 steps by `mtm continue-capped`. `pilot-haiku-v2` is the clean rerun before that, and
`pilot-haiku` predates the YAML defect fix (unquoted ` #`); keep both only for comparison, and do not re-analyze them with
current graders or rule text. `mtm validity` gives operator fidelity and the probe-suite analysis. Any rerun needs a new
`run_name`, because resume keys ignore rule text and graders and would keep old rows.

Next, in order:
1. By Nov 14: hand-label `labels_a.csv` using `labels_guide.md`, run `mtm label-summary`, add the result to the deck, then
   check the slides and rehearse. No new experiments before the talk.
2. Late Nov: small multi-rule pilot in a new run (5 to 10 rules per session, measured compression, agreed spend first).
   Go on only if it shows whether partial degradation appears.
3. Dec: systematic literature search, new probes mapped to rule clauses, second annotator, independent grader review.
4. Then: freeze the protocol and run the confirmatory held-out experiment, comparing mutation-guided, deletion-guided and
   random probe selection at equal budget, on at least two model families. Framed as an AI safety paper: safety
   evaluation and assurance of long-running enterprise operations agents, with coding and DevOps agents as a secondary
   domain (decided 2026-11-07). Later study compares compacted, pinned and independently enforced configurations. Venue still to choose; TOSEM and EMSE were the earlier targets.

## Conventions

- Graders must stay deterministic. Never use an LLM to decide whether a violation happened.
- Every new rule needs probes its grader flags; `mtm validate` and tests enforce this.
- Filler tasks must never violate any rule.
- Tools are simulated only; nothing is ever executed.
- Keep stages resumable (append JSONL, skip finished rows). One run directory holds one model; `run.json` enforces this.
- RQ4 splits must hold out whole rules, and its labels must come from replay.
