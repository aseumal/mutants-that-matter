"""Simulation-based power analysis for the confirmatory primary test (planning aid, not a result).

Estimand (PREREGISTRATION_DRAFT.md): Delta = U_M - U_D, the difference between mutation-guided (M) and
deletion-guided (D) selection in the share of held-out violation-inducing residues detected.
Primary weighting: by rule (every rule counts once: detection is averaged within each rule, then
across rules). Secondary: by residue. Both are reported.

Model, per simulated study:
- n_rules held-out rules; rule r has k_r >= 1 violation-inducing residues (Poisson mean k).
- Deletion detects a residue of rule r with probability p_D,r = expit(logit(b) + u_r), u_r ~ N(0, sd_b):
  baseline b with between-rule heterogeneity.
- Mutation detects with p_M,r = expit(logit(p_D,r) + theta + v_r), v_r ~ N(0, sd_e): a shift on the
  logit scale, so probabilities never need clipping. theta is calibrated by bisection so that the
  REALIZED mean of (p_M,r - p_D,r) over a large reference sample of rules equals the specified delta;
  cells where that is impossible (for example a +5 point gain when b = 0.97) are reported as infeasible.
- The two strategies test the SAME residues. Their detections are paired through a Gaussian copula
  with correlation rho (0 independent, 0.5, 0.9 nearly the same residues), giving the four joint
  outcomes: both, mutation only, deletion only, neither.
- Analysis: per-rule difference d_r = mean_M - mean_D over the rule's residues; estimate = mean of d_r
  (rule-weighted) with a t interval over rules; secondary estimate weights rules by k_r.
- Verdicts with margin m (mtm.stage2.verdict): benefit / equivalence / inferiority / inconclusive, and
  separately statistical superiority (lower limit above 0).
Nothing here is conservative by construction: dependence between strategies is varied, not assumed.

    python paper/analysis/power_sim.py     (from the repository root)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit

from mtm.stage2 import verdict

OUT = Path("paper/analysis/results/power.json")
SIMS, SEED, MARGIN = 1000, 20261128, 0.05
SD_BASE, SD_EFFECT = 0.8, 0.5          # logit-scale between-rule spread of baseline and of effect
GRID = {"b": (0.70, 0.85, 0.97), "k": (2, 6), "rules": (50, 80, 120),
        "delta": (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15), "rho": (0.0, 0.5, 0.9)}


def calibrate(b: float, delta: float) -> float | None:
    """theta whose realized mean(p_M - p_D) equals delta, on a fixed large reference sample."""
    ref = np.random.default_rng(1)
    u, v = ref.normal(0, SD_BASE, 200_000), ref.normal(0, SD_EFFECT, 200_000)
    p_d = expit(logit(b) + u)
    gap = lambda th: float(np.mean(expit(logit(p_d) + th + v) - p_d))
    lo, hi = -10.0, 10.0
    if not (gap(lo) <= delta <= gap(hi)):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < delta else (lo, mid)
    return (lo + hi) / 2


def one_study(n_rules, k_mean, b, theta, rho, rng):
    k = np.maximum(1, rng.poisson(k_mean, n_rules))
    p_d = expit(logit(b) + rng.normal(0, SD_BASE, n_rules))
    p_m = expit(logit(p_d) + theta + rng.normal(0, SD_EFFECT, n_rules))
    rule_of = np.repeat(np.arange(n_rules), k)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=len(rule_of))
    u = stats.norm.cdf(z)
    det_d = u[:, 0] < p_d[rule_of]
    det_m = u[:, 1] < p_m[rule_of]
    d = np.bincount(rule_of, det_m, n_rules) / k - np.bincount(rule_of, det_d, n_rules) / k
    t = stats.t.ppf(0.975, n_rules - 1)
    est, se = d.mean(), d.std(ddof=1) / np.sqrt(n_rules)
    w = k / k.sum()
    est_r = float((w * d).sum())
    se_r = float(np.sqrt((w ** 2 * (d - est_r) ** 2).sum() * n_rules / (n_rules - 1)))
    joint = (np.mean(det_d & det_m), np.mean(det_m & ~det_d), np.mean(det_d & ~det_m), np.mean(~det_d & ~det_m))
    return (est - t * se, est + t * se), (est_r - t * se_r, est_r + t * se_r), est, joint


def cell(n_rules, k_mean, b, delta, rho, rng):
    theta = calibrate(b, delta)
    if theta is None:
        return {"feasible": False}
    counts = {w: {"benefit": 0, "equivalence": 0, "inferiority": 0, "inconclusive": 0, "superiority": 0}
              for w in ("rule_weighted", "residue_weighted")}
    ests, joints = [], []
    for _ in range(SIMS):
        ci_rule, ci_res, est, joint = one_study(n_rules, k_mean, b, theta, rho, rng)
        for w, ci in (("rule_weighted", ci_rule), ("residue_weighted", ci_res)):
            counts[w][verdict(list(ci), MARGIN)] += 1
            counts[w]["superiority"] += ci[0] > 0
        ests.append(est)
        joints.append(joint)
    j = np.mean(joints, axis=0)
    out = {"feasible": True, "theta": round(theta, 3), "realized_mean_estimate": round(float(np.mean(ests)), 4),
           "joint_outcomes": dict(zip(("both", "mutation_only", "deletion_only", "neither"), np.round(j, 3).tolist()))}
    for w, c in counts.items():
        out[w] = {k: round(v / SIMS, 3) for k, v in c.items()}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = {}
    for b in GRID["b"]:
        for k in GRID["k"]:
            for n in GRID["rules"]:
                for delta in GRID["delta"]:
                    for rho in GRID["rho"]:
                        results[f"b={b:.2f} k={k} rules={n} delta={delta:+.2f} rho={rho}"] = cell(n, k, b, delta, rho, rng)
    out = {"settings": {"simulations_per_cell": SIMS, "margin": MARGIN, "sd_base_logit": SD_BASE,
                        "sd_effect_logit": SD_EFFECT, "grid": GRID}, "cells": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for key, c in results.items():
        if not c["feasible"]:
            print(f"{key:44s} infeasible (cannot realize this delta at this baseline)")
            continue
        r = c["rule_weighted"]
        print(f"{key:44s} realized {c['realized_mean_estimate']:+.3f}  benefit {r['benefit']:.2f}  "
              f"superiority {r['superiority']:.2f}  equivalence {r['equivalence']:.2f}  inferiority {r['inferiority']:.2f}")


if __name__ == "__main__":
    main()
