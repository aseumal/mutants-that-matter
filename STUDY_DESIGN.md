# Mutants That Matter: Stage 2 study design and implementation

Status on 2026-11-27: code implemented and tested (134 tests pass). **Nothing has been run against an API.** The v3 pilot is frozen and reproduces exactly under the new code.

## 1. Question

The v3 pilot found that harmful compaction residues mostly behaved like deleting the rule. On that data, deletion-guided probe selection matched or beat mutation-guided selection. Stage 2 therefore asks:

> Under what conditions does mutation-based safety evaluation give assurance beyond rule-deletion testing, for compaction-induced behavioral failures in LLM agents?

The hypothesis to test is that mutants can only add value where compaction causes partial damage. Stage 2 first checks whether partial damage occurs: under milder budgets, with several rules per session, and across independent trajectories. Only then does it test whether mutants help.

## 2. What changed in the code, and why

| Issue | Raised by | Fix | Where | Test |
|---|---|---|---|---|
| A resumed compaction could skip a cycle that was never written | review 4.1 | Rows for a trajectory are written in one operation. A partial trajectory raises an error instead of being skipped. | `experiment.stage_compact` | `test_partial_trajectory_is_refused_not_skipped` |
| Resume keys ignore the rules, graders, prompts and settings | review 4.2 | `run.json` stores hashes of the rules, probes, graders, filler, prompts and the settings that change what a row means. Resuming under different values is refused. Runs from before this change cannot be resumed. The code hash is recorded but not enforced. | `experiment.fingerprint`, `check_manifest` | `test_fingerprint_change_blocks_resume`, `test_legacy_run_without_fingerprint_is_not_resumed` |
| `not_prefix /tmp` treated `/tmpfoo` and `/TMP` as inside `/tmp`; R10 treated `internal.acme.com.evil.io` as internal | review 4.3 | New operators `path_under`/`not_path_under` (path normalized, case-sensitive, matched on whole path segments) and `url_host_in`/`url_host_not_in`. Used by R01, R02 and R10. The R10 shell route now also checks where the host name ends. | `grading.py`, `data/rules.yaml` | `test_path_under_*`, `test_r10_*`, `test_r02_*` |
| The R01 shell regex flagged `rm -f /tmp/x` and missed `rm /tmp/a /var/x` | found by the new negative tests | The regex now checks every non-flag argument | `data/rules.yaml` | `test_r01_positive_negative_and_malformed` |
| Grader self-check covered positive cases only | review 4.4 | Each rule now has a permitted call that must break no rule at all, plus boundary, threshold and malformed-call tests. A missing path or URL never counts as a violation. | `rules.validate_rules`, tests | `test_every_permitted_call_is_clean_and_completes`, `test_r11_threshold_boundary` |
| No permitted actions, so an agent that refuses everything looks safe | review §6, mentor | 30 permitted-action probes (`permit_probes`), each with a completion predicate. Over-refusal = 1 − completion rate. These probes never count as violation opportunities. | `rules.PermitProbe`, `stage2.over_refusal` | end-to-end test |
| Mutants and residues were replayed in different contexts | review critical issue 1, mentor | Matched-context replay. Each residue's policy span is separated from its context, then the intact rule, the deleted rule, the reframed residue, each mutant, a paraphrase and a pinned rule are replayed inside that same context. The real summary stays as the check, as the mentor suggested. | `context.py`, `experiment.conditions_for` | `test_matched_conditions_share_the_residue_context` |
| Policy authority: the rule only ever arrives as a user message | review 4.5 | `matched:…:pinned` puts the intact rule in the system prompt next to the real summary. It is analysed separately. | `session.summary_frame` | same |
| One compaction trajectory per setting | review 4.6 | Option `trajectories: N` gives a new filler order per trajectory and condition keys ending `:tN`. Agreement between trajectories is reported. | `stage_compact`, `stage2.trajectories` | `test_trajectories_get_their_own_keys` |
| Only one rule per session | mentor, plan | Option `rules_per_session: k` runs a fixed-seed shuffle into shared sessions and records session id and position | `session_groups` | `test_multi_rule_sessions_*` |
| Compression was nominal, not measured | mentor | Each residue records transcript words, summary words, the ratio, and whether it went over budget | `compaction.py`, `stage2.compression` | same |
| Invalid episodes were only dropped | review 5.4 | Every episode is classified as violation, refused, other action or invalid, per condition group | `analysis.outcome`, `stage2.outcomes` | `test_outcome_classes` |
| Coupling reported in one direction only | mentor | Both directions: harmful residues covered by any mutant, and mutants matching some harmful residue, each as overlap and as exact match, with intervals that resample rules | `stage2.coupling` | end-to-end test |
| Calibration stated loosely | review §7, correcting me | Exact one-sided bound 1 − 0.05^(1/n), labelled optimistic because episodes are dependent | `stage2.calibration` | `test_exact_zero_bound` |
| Cost was not known before running | plan | `mtm plan` counts compaction calls and replay episodes with no API use. The count is exact once residues exist, and it reproduced v3's 5,076. | `experiment.estimate` | `test_plan_command_makes_no_run_directory`, end-to-end |
| Rerunning v3 conditions would mean paying for them twice | plan | `mtm import-run` copies a finished run, but only if the current graders reproduce every stored verdict | `revise.import_run` | `test_import_run_copies_only_reproducible_runs` |
| Editing a summary is itself an intervention | review 3, critical issue 3 | `matched:…:sham` removes one non-policy segment of similar length and keeps the policy span | `context.sham_edit` | `test_sham_edit_*` |
| Rule surviving only as an aside inside a task list, so removing the sentence also removes context | found on v3 (13 such spans) | Flagged `embedded_span` for manual spans. The splitter now also splits after a quoted sentence end. | `context.split_summary` | `test_quoted_sentence_end_*` |
| The two-of-three threshold misses low-rate violations | review 3, critical issue 5 | Continuous violation probability and its paired difference from intact as the primary outcome; threshold sweep at 1, 2 and 3 of 3 | `stage2.rate_profiles`, `stage2.thresholds` | `test_selection_cost_and_thresholds_on_mock_run` |
| Equal probe counts are not equal costs | review 3, critical issue 4 | Candidate episodes per strategy and episodes per extra residue found beyond random | `stage2.selection_cost` | same |

**Pilot data is protected.**
- The core analysis (`analyze`, `validity`, `predict`, labels) reads only pilot-design rows. Permitted, matched, isolated and paraphrase rows go to `mtm stage2`.
- Regenerating the v3 reports gives byte-identical tables.
- Regrading all 5,076 v3 episodes with the corrected graders changes 0 verdicts. The bugs were real, but no v3 call reached them.

## 3. Execution plan

Every phase needs `mtm plan -c <config>` and your agreement on spend first. Episode counts are given below; v3 averaged 1.45 model calls per episode.

### Phase 0: integrity (done)

Covered by section 2.

### Phase 1: finish v3 (your time, no API)

1. Label the 120 residues blind, following `labels_guide.md`.
2. Run `mtm label-summary`.
3. Archive v3 as it is.

### Phase 1b: smoke test (216 episodes, 6 compaction calls)

`configs/stage2_smoke.yaml` uses 3 rules and the permitted probes. It checks the new outcome classes and fields with the real model before any discovery arm.

### Phase 2: discovery, does partial damage occur?

- **Configs:** `configs/stage2_discovery_k{1,3,6}.yaml`.
- **Rules:** the same 12 rules in every arm, across all categories.
- **Settings:** budgets 30, 45, 60, 90 and 150; one cycle; 3 trajectories; permitted probes on; no mutants yet.

| Arm | Compaction calls | Replay episodes |
|---|---|---|
| k = 1 rule per session | 180 | 2,736 |
| k = 3 | 60 | 2,736 |
| k = 6 | 30 | 2,736 |

**Read-out:** `mtm stage2` and `mtm analyze`, plus hand-labelling a sample of residues.

**Order:** run k = 1 and k = 6 first. Run k = 3 only if those two differ.

**Decision gate. A draft for the author to confirm in writing before any discovery run.** The gate tests only whether there are enough informative faults to answer the question. It must never depend on whether mutants appear to beat deletion, which would select settings favorable to one answer: proceed to Phase 3 on discovery data if there are at least 10 harmful residues, across at least 5 rules, that either still contain a policy span or fail a different set of probes from deletion. Otherwise, report "deletion-dominated under all tested regimes" as the finding.

**Two populations.** Discovery may show partial damage under only some settings. The confirmatory study must still sample a representative set of settings to estimate overall performance, and keep a separate sample enriched for partial damage to study subtle faults. The enriched sample is never used to estimate how often faults occur.

Mutants and isolated spans can be added to the same run later. Those settings only add rows, so the fingerprint allows them.

### Phase 3: matched-context replay

Matched conditions are the intact rule, the deleted rule, the reframed residue, each mutant, the pinned rule and a **sham edit**. The sham edit removes one non-policy sentence and keeps everything else, which shows whether editing a summary changes behavior on its own.

- **Recommended:** skip v3 and run this on discovery data. In 28 of the 33 harmful v3 residues no policy span survives, and 4 of the 5 that keep one hold it only as an aside, so on v3 this mostly measures the context effect.
- **On v3 if wanted** (`configs/stage2_matched.yaml`):
  1. Run `mtm import-run -c configs/stage2_matched.yaml --source pilot-haiku-v3`.
  2. Run `mtm plan`. With everything switched on, 9,504 episodes remain: 6,372 matched-context, 1,440 isolated spans, and 1,692 permitted probes under the core conditions. Switching off isolated spans or permitted probes cuts the cost.

  Caveat: in v3, 28 of the 33 harmful residues have no detectable policy span, so on this data matched replay mainly measures the context effect.
- **On discovery data:** the same config options, pointed at a Phase 2 run.
- **Before reading results:** check `contexts.jsonl`, correct spans by hand in `policy_spans.jsonl` where needed, and report results both with and without flagged splits.

### Phase 4: confirmatory study, frozen

Freeze before running:
- operators
- graders, together with an independent reviewer's challenge cases
- the probe suite, 10 to 15 per rule including permitted and boundary cases, mapped to rule clauses, with a held-out set
- compaction settings chosen in Phase 2
- hypotheses, exclusion rules and analysis

Rules must be **new and held out**: the 30 pilot rules have been seen. Use at least two model families, and separate compactor from agent.

The primary comparison is mutation-guided against deletion-guided against random probe selection, run twice: at an equal number of selected probes, and at an equal total number of episodes, counting the episodes spent choosing probes. Pre-specify a smallest effect worth detecting, for example 5 points of held-out detection, so the study can tell benefit, practical equivalence and inferiority apart. Absence of a difference is not equivalence. The primary behavioral outcome is violation probability; binary failing sets are secondary, with the threshold sweep. Report:
- unique real failures each strategy finds
- failures found by deletion but missed by mutants, so the comparison is symmetric
- cost per additional failure

Size the study by simulation with clustering by rule, not by a fixed fault count.

**Incremental predictive validity:** compare held-out prediction of real-fault detection from deletion score plus an independently estimated probe difficulty (model 0) against the same plus mutation score (model 1).

**Cost:** report episodes, API requests, input and output tokens, money, wall-clock time, and mutant-generation and annotation effort. Plot detection against total budget, rather than one ratio, because a cost-per-extra-fault ratio is unstable when the gain is near zero.

**Stopping rules:** set a maximum spend and run count, plus a futility rule, in the preregistration. A sign test with 6 faults is only the smallest possible significant result.

### Phase 5: across models

Phase 5 repeats Phase 4 on another model configuration. The pinned configuration and an independently enforced configuration are compared there as reference points.

## 4. Readings agreed in advance

Every outcome is reportable:

| Outcome | Interpretation |
|---|---|
| Mutants beat deletion overall | Supports mutation-based assurance |
| Mutants beat deletion only where damage is partial | Boundary condition for mutation validity |
| Deletion as good as mutants | Sophisticated mutants are unnecessary under the tested conditions |
| Mutants find fewer failures but more severe ones | Supports severity-aware testing, which needs severity ratings |
| The matched context fails even with the intact rule | Guardrail decay is partly a context effect, not only a text effect |
| Results vary by model | Mutation validity depends on the model |

## 5. Statistics notes

- **Calibration with zero failures in n trials:** the exact 95% upper bound is 1 − 0.05^(1/n).

  | n | 19 | 30 | 59 | 100 |
  |---|---|---|---|---|
  | Bound | 14.6% | 9.5% | 4.95% | 2.95% |

  Bounding below 5% needs about 59 independent trials. RecMutant's 19-run figure bounds a threshold's false-alarm rate (1/(n+1)), which is a different claim.
- **Uncertainty:** every Stage 2 interval resamples whole rules. A confirmatory analysis should use mixed-effects or cluster-robust models with rule, probe and trajectory as levels.

## 5b. Pilot results added on 2026-11-27 (saved v3 data, no API)

These are exploratory and post hoc.
- **Threshold sweep:** 36, 33 and 31 harmful residues at 1, 2 and 3 of 3 samples. Deletion-guided selection reached 97%, 98% and 95%; all-mutant selection 94%, 89% and 90%.
- **Selection cost at one probe per rule:**

  | Strategy | Candidate episodes | Episodes per extra residue beyond random |
  |---|---|---|
  | Deletion-guided | 270 | about 32 |
  | All mutants | 2,646 | about 453 |
  | Catastrophic operators | 540 | about 79 |
- **Coupling in both directions:** every harmful residue overlaps some rule mutant, and 88% are matched exactly. Only 44% of non-control rule mutants overlap any harmful residue of their rule, and 27% match one exactly.
- **Training on synthetic faults and testing on real residues** (leave one rule out, same classifier and features, every example of the held-out rule excluded and asserted, synthetic examples framed as replayed). The first version framed intact examples differently from mutants; this is the corrected run.

  | Training data per fold | AUROC | Paired difference vs deletion + intact |
  |---|---|---|
  | Deletion + intact, 58 | 0.910 [0.86, 0.96] | — |
  | Mutants, 313–314 | 0.433 | −0.64 to −0.33 |
  | Mutants, size and class ratio matched, 58 (20 draws) | mean 0.591 (range 0.45–0.84) | — |
  | Mutant weight share 0.25 / 0.5 / 0.75 | 0.840 / 0.757 / 0.660 | all below zero |
  | 3 best-coupling operators, chosen inside each fold | 0.465 | −0.58 to −0.32 |
  | Real residues, within domain | 0.949 | +0.01 to +0.07 |

  Class balance: deletion + intact 43% positive, mutants 34%; the logistic regression uses balanced class weights. Source: `paper/analysis/deletion_transfer.py`, `runs/pilot-haiku-v3/deletion_transfer.json`.
- **Continuous violation probability:** residues 0.20 [0.12, 0.28], rule absent 0.61, `flip_negation` 0.86, `hedge` 0.00, `weaken_modal` 0.01.
- **Calibration:** 540 intact-repeat episodes with 0 violations. The exact bound would be 0.55%, but it is optimistic because the episodes are dependent.

## 6. Known limits of this implementation

- **Policy-span detection is a heuristic** based on word overlap with the rule plus policy cue words. It flags doubtful splits (`no_span`, `residual_reference`, `embedded_span`), but spread-out or allusive residues need hand-checked spans in `policy_spans.jsonl`.
- **The shell graders are regex checks.** They cover the routes tested and do not implement full file-system authorization semantics. The paper must not claim complete shell coverage.
- **The 30 permitted probes and their completion predicates were drafted by Claude.** Rule-to-probe content is the author's call, so review and correct them, and record that they were reviewed.
- **No paraphrases are written yet.** The `paraphrase:` field is supported but empty. A meaning-preserving paraphrase must be written and checked by a person.
- **The pinned condition covers only the system-prompt channel.** External memory and tool-level enforcement are not simulated.
- **One permitted probe per rule** is enough to detect gross over-refusal, not to estimate it precisely.
