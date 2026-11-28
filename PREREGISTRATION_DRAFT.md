# Confirmatory study protocol: DRAFT for the author to confirm

Mutants That Matter, confirmatory stage. Prepared on 2026-11-28 from the pilot and its reviews.

**Nothing in this document has been run.** Every number marked *proposed* is a choice for the author (and mentor) to confirm or change before freezing. Once frozen, the document is time-stamped and archived before any confirmatory data exists.

## 1. Hypotheses
| ID | Hypothesis | Outcome |
|---|---|---|
| **H1 (primary)** | Mutation-guided probe selection detects meaningfully more held-out violation-inducing residues than deletion-guided selection at equal total cost | Δ(B), below |
| H2 | Mutation fidelity depends on the damage mechanism | Behavioral similarity by annotated damage type |
| H3 | Matched-context replay changes the estimated fidelity of mutants | Paired differences in violation probability, neutral against matched frame |
| H4 | Closer alignment between mutant and residue damage distributions improves predictive transfer | Held-out AUROC/AUPRC against Jensen–Shannon divergence of damage types |

H1 stays primary whatever H2–H4 show. Not supporting H1 is a reportable result.

## 2. Primary estimand (one operational definition, frozen)
For held-out violation-inducing residues R, strategy j ∈ {M (mutation-guided), D (deletion-guided)}, total budget B per rule, and the suite T_j(B) that j selects:

- **Detection** *(proposed)*: residue ρ is detected by T_j(B) if at least one of s = 10 replays of any probe in T_j(B) violates the rule. This is empirical detection, matching how a team would test. Probability-weighted detection, d_j(ρ) = 1 − ∏_q (1 − p̂_ρq)^s, is a secondary outcome.
- **Utility (primary: rule-weighted)**: U_j(B) is the share of a rule's held-out violation-inducing residues that are detected, averaged over rules, so every rule counts once. **Secondary: residue-weighted**, where every residue counts once. The simulation, the manuscript and the final analysis all use this same weighting.
- **Joint outcomes**: because both strategies test the same residues, every residue is classified as detected by both, mutation only, deletion only, or neither. The discordant counts (mutation only, deletion only) are reported with Δ, so complementarity stays visible when Δ is near zero.
- **Primary contrast**: Δ(B) = U_M(B) − U_D(B), with a 95% interval clustered by rule, and trajectory nested within rule.
- **Equal total cost**: B counts every replay episode a strategy spends, including those used to choose its probes. Strategy j spends C_j = (conditions_j × candidate probes × s_sel) on selection and uses B − C_j to test residues. The number of probes it can test is fixed by its remaining budget, so expensive selection buys fewer test probes. A secondary analysis compares strategies at an equal number of probes.
- **Verdict** (`mtm.stage2.verdict`), with margin m:

  | Interval position | Verdict |
  |---|---|
  | entirely above +m | **benefit** |
  | entirely below −m | **inferiority** |
  | entirely inside [−m, +m] | **equivalence** |
  | anything else | **inconclusive** |

  Statistical superiority (interval above 0) is reported separately and never called meaningful.
- **Margin** *(proposed m = 5 points; needs a justification the author agrees with)*. A draft justification: missing 5 more violation-inducing compaction faults per 100 is the smallest loss a team would trade for the 2–10 times higher selection cost the pilot measured. Raise m if missing a single high-severity failure is unacceptable in the target deployment, and pair it with severity weighting.

## 3. Probe partitions
| Population | Written by | Used for | Never used for |
|---|---|---|---|
| Development | author (the 90 pilot probes) | building operators, graders and analyses | any confirmatory number |
| Held-out candidates | author, before seeing held-out residues; 10–15 per held-out rule | the pool each frozen strategy selects from | defining which residues are violation-inducing |
| Independent reference | a second person, blind to candidates and operators; at least 5 per held-out rule | defining which held-out residues are violation-inducing (the fault labels) | selection |

Separating candidates from reference probes stops the definition of a fault from depending on the probes being judged. That was the shared-yardstick problem in the pilot, and the oracle-dependence lesson from RecMutant.

## 4. Faults and rules
- **Held-out rules**: new rule families or the coding/DevOps domain (`data/heldout_rules.yaml`, checked by `mtm validate --rules`). Not rewordings of pilot rules.
- **Held-out faults**: residues from independent compaction trajectories (new filler order and seeds), generated after the protocol is frozen.
- **Discovery gate**: decided on the supply of independently annotated partial faults, and on adequate measurement, never on how mutants or deletion perform. *Proposed*: at least 10 violation-inducing residues across at least 5 rules that annotators judge partially preserved, with grader and span reviews passed. There is **no condition on deletion's detection rate**. A gate such as "deletion detects at most 75%" would select settings where the comparator is weak, and is withdrawn.
- **If deletion dominates** (detection near the ceiling): the study still runs and reports the ceiling, equivalence or inferiority, the discordant outcomes (complementarity) and cost. Strong evidence that the simpler method suffices is a result, not a failure.
- **Two populations**:
  - *Representative*: the settings sampled with fixed weights. It estimates overall utility.
  - *Enriched*: partial-damage residues. It answers H1 where mutants could matter.

  The enriched sample is never used to estimate how often faults occur.

## 5. Sample size (simulation, `paper/analysis/power_sim.py`; planning scenarios, not frozen)
- **Model.** The two strategies' outcomes on each residue are drawn jointly: both detect, mutation only, deletion only, neither.
- **Grid.**
  - dependence between strategies ρ = 0, 0.5, 0.9 (a Gaussian copula)
  - deletion baseline b = 0.70, 0.85, 0.97, with between-rule heterogeneity
  - 2 or 6 residues per rule
  - 50, 80 or 120 rules
  - true Δ from −10 to +15 points
- **Calibration.** Effects are shifts on the logit scale, calibrated so the realized mean effect equals the stated Δ. Cells where Δ cannot be realized are marked infeasible.
- **Estimand.** Rule-weighted primary; residue-weighted reported in `results/power.json`.
- **Runs.** 1,000 simulated studies per cell.

| b, residues per rule, rules | True Δ | P(meaningful benefit), ρ = 0 / 0.5 / 0.9 | P(superiority), ρ = 0 / 0.5 / 0.9 |
|---|---|---|---|
| 0.70, 6, 80 | +10 | 0.36 / 0.46 / 0.69 | 0.87 / 0.96 / 1.00 |
| 0.70, 6, 80 | +15 | 0.90 / 0.96 / 1.00 | 1.00 |
| 0.70, 6, 120 | +10 | 0.50 / 0.61 / 0.83 | 0.98 / 0.99 / 1.00 |
| 0.70, 6, 120 | 0 | P(equivalence) 0.01 / 0.17 / 0.64 | P(false superiority) ≈ 0.02–0.03 |

| b, residues per rule, rules | True Δ | P(inferiority), ρ = 0.5 |
|---|---|---|
| 0.70 / 0.85 / 0.97, 6, 120 | −10 | 0.55 / 0.68 / 0.87 |

| b, residues per rule, rules | True Δ | P(equivalence), ρ = 0 / 0.5 / 0.9 |
|---|---|---|
| 0.97, 6, 120 | 0 | 0.97 / 0.99 / 1.00 |
| 0.97, any | ≥ +5 | infeasible (ceiling) |

What this means:
1. **Dependence matters.** A design whose strategies agree on most residues (high ρ) has much more power. Dependence cannot be assumed, so it must be estimated in the feasibility and discovery runs before the size is frozen.
2. **In a deletion-dominated population no gain of 5 points or more is possible,** but the study can establish equivalence with high probability.
3. **80–120 held-out rules with 6 residues each are planning scenarios, not a frozen size.** Re-run the simulation with discovery estimates of b, ρ and between-rule spread before freezing.

## 6. Models, sampling and controls
- **Models**: at least two families, with compactor and agent varied separately. At least one family is open-weight (for constrained-action likelihood diagnostics only; repeated sampling stays the primary measure).
- **Samples**: balanced across arms. Every condition in a comparison gets the same number of samples (10 *proposed*). Unequal allocation is allowed only if preregistered and simulated.
- **Controls**:
  - meaning-preserving paraphrase (author-written, checked)
  - sham edit
  - pinned rule (reference)
  - `drop_scope` (broadening)
  - random selection
  - planted mutants in real context, read as context dependence (a mismatch with the neutral frame is a finding, not a probe failure)

## 7. Stopping and spending *(proposed)*
- A spending cap per phase, agreed in writing before each phase, using `mtm plan` estimates.
- **Futility look after half the held-out rules.** Stop early only if the completed study is unlikely to give **any** informative verdict: the conditional probabilities of benefit, equivalence and inferiority are all below 10%. Low odds of mutation superiority alone is **not** a reason to stop, because establishing equivalence or inferiority, that the simpler method suffices, is also valuable.
- **No early stopping for benefit.** The spending cap still applies.
- Exclusions, defined before data:
  - invalid episodes (reported per arm)
  - rules whose graders fail independent review
  - residues whose policy-span edits fail review

## 8. Independent verification (before confirmatory claims)
- [ ] A person other than the tool author reviews every grader, with challenge cases
- [ ] A stratified sample of raw episodes is read and checked against stored verdicts
- [ ] Primary results are recomputed with a separate analysis script
- [ ] Rule-level separation of train and test is checked in every classifier analysis
- [ ] Each cited paper is checked against its primary source
- [ ] Archive hashes are verified against the released data
- [ ] Residues are annotated by two blinded annotators, with agreement measured before adjudication
- [ ] Policy-span edits are reviewed by a person for meaning, not only word-for-word reconstruction
- [ ] Final manuscript claims are checked line by line against the results

## 9. Order of work
1. Finish the pilot: labels, grader review, shortcut diagnostics (done), manuscript corrections.
2. Feasibility: a 12-residue matched-context replication (`configs/stage2_feasibility.yaml`, `mtm import-run --fresh`), 2,544 episodes, or 2,832 with the 12 paraphrases. Natural and neutral conditions are replayed fresh. Not confirmatory.
   - Runs in three batches with a required spending cap: 1 rule (240 episodes), 4 rules (852), 7 rules (1,452).
   - Fresh and historical rows carry separate `origin` tags. They are compared as a check on drift between runs, never pooled.
   - Human review first: spans, paraphrases and grader challenge cases (`docs/review/`).
   - **Must hold, or the run is repeated after a fix:**
     - complete records (accounting: planned = executed + reported unavailable)
     - graders behave as reviewed
     - every planned condition was generated, or is listed as unavailable with its reason
     - every edit passed human span review and changed only the intended rule
     - invalid and censored episodes stay within a tolerance agreed beforehand (*proposed*: ≤ 2% of episodes, with each one explained)
   - **Empirical findings, never failures:**
     - a sham edit that changes behavior (context sensitivity)
     - deletion failing to reproduce the natural failures
     - planted or appended mutants behaving differently from the neutral frame
     - a pinned rule that still sees violations
     - fresh natural replays that differ from v3 (drift between runs)
   - **Report:** paired continuous violation-probability differences against the natural residue, and the strategies' dependence (ρ) for the size re-run.
3. Discovery: k = 1 and k = 6 arms, then k = 3 if they differ. The gate is on fault supply.
4. Freeze this protocol: archive it with a timestamp and hash.
5. Confirmatory run on held-out rules and trajectories, with a futility look at half.
6. Cross-model replication.
