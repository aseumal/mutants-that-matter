"""Exploratory: can a trainable representation transfer from synthetic mutants to real residues?

Every model trains on synthetic examples from the other rules only, and is scored on the
real compaction residues of the held-out rule (leave one rule out, 30 folds). Labels come
from replay. All settings below were fixed before the first run; there is no validation
data, so nothing is tuned. Written on 2026-11-13 after the pilot results were known, so
every result here is exploratory.

    python paper/analysis/dl_transfer.py            (from the repository root)
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mtm import predict as P
from mtm.experiment import NEUTRAL_SUMMARY
from mtm.rules import load_rules
from mtm.validity import Run

RUN = Path("runs/pilot-haiku-v3")
SEED = 7
HEAD = dict(hidden=256, proj=64, temperature=0.1, lam=0.5, lr=1e-3, weight_decay=1e-4, epochs=300)
FINETUNE = dict(lr_encoder=2e-5, lr_head=1e-3, epochs=4, batch=32)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Item:
    rule_id: str
    rule: str
    text: str
    label: int


def load() -> tuple[list[Item], list[Item], list[Item]]:
    rules = {r.id: r for r in load_rules()}
    residues, mutants = P.load_examples(RUN, rules)
    run = Run(RUN)
    to_item = lambda e: Item(e.rule_id, e.rule, e.text, int(e.label))
    absent = [Item(r, rules[r].text, NEUTRAL_SUMMARY, int(bool(run.fails.get((r, "absent")))))
              for r in sorted(rules)]
    return [to_item(e) for e in residues], [to_item(e) for e in mutants], absent


# ------------------------------------------------------------------ shared pair features

def closest_sentences(items: list[Item], model) -> list[str]:
    """The summary sentence most similar to the rule, picked once with the frozen encoder."""
    out = []
    for it in items:
        sents = P.sentences(it.text)
        vecs = model.encode([it.rule] + sents, normalize_embeddings=True, show_progress_bar=False)
        out.append(sents[int(np.argmax(vecs[1:] @ vecs[0]))])
    return out


def pair_features(u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.cat([u, s, (u - s).abs(), u * s, (u * s).sum(-1, keepdim=True)], dim=-1)


class Head(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(dim, HEAD["hidden"]), torch.nn.ReLU(),
                                        torch.nn.Linear(HEAD["hidden"], HEAD["proj"]))
        self.clf = torch.nn.Linear(HEAD["proj"], 1)

    def forward(self, x):
        z = F.normalize(self.body(x), dim=-1)
        return z, self.clf(z).squeeze(-1)


def supcon(z: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020): same-label examples pulled together."""
    sim = z @ z.T / t
    self_mask = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.any(1)
    if not has_pos.any():
        return z.sum() * 0
    return -((log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()


def loss_fn(z, logits, y, lam):
    pos_weight = torch.tensor((len(y) - y.sum()) / max(1.0, float(y.sum())))
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    return bce + lam * supcon(z, y, HEAD["temperature"]) if lam else bce


# ------------------------------------------------------------------ models

def head_scores(x_tr, y_tr, x_te, lam: float, seed: int) -> np.ndarray:
    """A trainable projection head on frozen MiniLM pair features."""
    seed_all(seed)
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt = torch.tensor((x_tr - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    head = Head(xt.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=HEAD["lr"], weight_decay=HEAD["weight_decay"])
    for _ in range(HEAD["epochs"]):
        opt.zero_grad()
        z, logits = head(xt)
        loss_fn(z, logits, yt, lam).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor((x_te - mean) / std, dtype=torch.float32))[1]).numpy()


def finetune_scores(model_name, tr: list[tuple[str, str]], y_tr, te: list[tuple[str, str]], seed: int) -> np.ndarray:
    """Unfreeze MiniLM's last transformer layer, trained jointly with the contrastive head."""
    from sentence_transformers import SentenceTransformer

    seed_all(seed)
    st = SentenceTransformer(model_name, device="cpu")
    encoder = st[0].auto_model
    for p in encoder.parameters():
        p.requires_grad = False
    last = encoder.encoder.layer[-1]
    for p in last.parameters():
        p.requires_grad = True

    def embed(texts: list[str]) -> torch.Tensor:
        feats = st.tokenize(texts)
        out = st[0]({k: v for k, v in feats.items()})
        out = st[1](out)
        return F.normalize(out["sentence_embedding"], dim=-1)

    head = Head(4 * 384 + 1)
    opt = torch.optim.Adam([{"params": last.parameters(), "lr": FINETUNE["lr_encoder"]},
                            {"params": head.parameters(), "lr": FINETUNE["lr_head"]}],
                           weight_decay=HEAD["weight_decay"])
    y_all = torch.tensor(y_tr, dtype=torch.float32)
    order = list(range(len(tr)))
    st.train()
    for _ in range(FINETUNE["epochs"]):
        random.shuffle(order)
        for i in range(0, len(order), FINETUNE["batch"]):
            idx = order[i:i + FINETUNE["batch"]]
            u = embed([tr[j][0] for j in idx])
            s = embed([tr[j][1] for j in idx])
            z, logits = head(pair_features(u, s))
            opt.zero_grad()
            loss_fn(z, logits, y_all[idx], HEAD["lam"]).backward()
            opt.step()
    st.eval()
    with torch.no_grad():
        u = embed([a for a, _ in te])
        s = embed([b for _, b in te])
        return torch.sigmoid(head(pair_features(u, s))[1]).numpy()


# ------------------------------------------------------------------ experiment

def main(do_finetune: bool = True) -> None:
    residues, mutants, absent = load()
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(P.ENCODER_NAME, device="cpu")
    encode = lambda texts: np.asarray(st.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def frozen_pairs(items):
        s = closest_sentences(items, st)
        u, v = encode([i.rule for i in items]), encode(s)
        return pair_features(torch.tensor(u), torch.tensor(v)).numpy(), s

    x_res, s_res = frozen_pairs(residues)
    x_mut, s_mut = frozen_pairs(mutants)
    x_abs, s_abs = frozen_pairs(absent)
    # The existing RQ4 features, to reproduce the recorded mutant-only logistic regression.
    px_res, _ = P.featurize([P.Example(i.rule_id, i.rule, i.text, bool(i.label), 0.0) for i in residues], encode)
    px_mut, _ = P.featurize([P.Example(i.rule_id, i.rule, i.text, bool(i.label), 0.0) for i in mutants], encode)
    px_abs, _ = P.featurize([P.Example(i.rule_id, i.rule, i.text, bool(i.label), 0.0) for i in absent], encode)

    y = np.array([i.label for i in residues])
    groups = [i.rule_id for i in residues]
    rules = sorted(set(groups))
    scores: dict[str, np.ndarray] = {}

    def run(name: str, fit, with_absent: bool):
        out = np.zeros(len(residues))
        for r in rules:
            tr = [k for k, m in enumerate(mutants) if m.rule_id != r]
            ta = [k for k, a in enumerate(absent) if a.rule_id != r] if with_absent else []
            te = [k for k, it in enumerate(residues) if it.rule_id == r]
            assert all(mutants[k].rule_id != r for k in tr) and all(absent[k].rule_id != r for k in ta)
            out[te] = fit(tr, ta, te)
        scores[name] = out
        print(f"  {name}: AUROC {roc_auc_score(y, out):.3f}", flush=True)

    ymut = np.array([m.label for m in mutants])
    yabs = np.array([a.label for a in absent])
    stack = lambda a, b, ia, ib: np.vstack([a[ia], b[ib]]) if len(ib) else a[ia]
    ycat = lambda ia, ib: np.concatenate([ymut[ia], yabs[ib]])

    print("frozen encoder, trained on synthetic data from other rules:", flush=True)
    for with_absent, tag in ((False, "mutants"), (True, "mutants + deletion")):
        run(f"logreg | {tag}", lambda tr, ta, te: P._fit_score(
            "logreg", stack(px_mut, px_abs, tr, ta), ycat(tr, ta), px_res[te], SEED), with_absent)
        run(f"mlp | {tag}", lambda tr, ta, te: P._fit_score(
            "mlp", stack(px_mut, px_abs, tr, ta), ycat(tr, ta), px_res[te], SEED), with_absent)
        for seed in (SEED, SEED + 1, SEED + 2):
            run(f"contrastive head s{seed} | {tag}", lambda tr, ta, te, seed=seed: head_scores(
                stack(x_mut, x_abs, tr, ta), ycat(tr, ta), x_res[te], HEAD["lam"], seed), with_absent)
            run(f"same head, no contrastive loss s{seed} | {tag}", lambda tr, ta, te, seed=seed: head_scores(
                stack(x_mut, x_abs, tr, ta), ycat(tr, ta), x_res[te], 0.0, seed), with_absent)

    if do_finetune:
        print("last MiniLM layer fine-tuned with the contrastive head:", flush=True)
        pairs_mut = [(m.rule, s) for m, s in zip(mutants, s_mut)]
        pairs_abs = [(a.rule, s) for a, s in zip(absent, s_abs)]
        pairs_res = [(i.rule, s) for i, s in zip(residues, s_res)]
        for with_absent, tag in ((False, "mutants"), (True, "mutants + deletion")):
            run(f"fine-tuned last layer + contrastive | {tag}", lambda tr, ta, te: finetune_scores(
                P.ENCODER_NAME, [pairs_mut[k] for k in tr] + [pairs_abs[k] for k in ta],
                ycat(tr, ta), [pairs_res[k] for k in te], SEED), with_absent)

    # Report, with intervals that resample rules, and the subset that is not deletion-like.
    run_obj = Run(RUN)
    not_del = np.array([bool(i.label) and run_obj.fails.get((i.rule_id, c)) != run_obj.fails.get((i.rule_id, "absent"))
                        for i, c in zip(residues, [r["condition"] for r in json.loads("[" + ",".join(
                            (RUN / "residues.jsonl").read_text().strip().splitlines()) + "]")])])
    keep = (y == 0) | not_del
    report = {"settings": {"head": HEAD, "finetune": FINETUNE, "seed": SEED},
              "residues": len(y), "harmful": int(y.sum()), "not_deletion_like_harmful": int(not_del.sum()),
              "models": {}}
    for name, s in scores.items():
        m = P.metrics(y, s, groups, seed=SEED)
        m["auroc_not_deletion_like"] = round(float(roc_auc_score(y[keep], s[keep])), 3) if not_del.any() else None
        report["models"][name] = m
    Path(RUN / "dl_transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (v["auroc"], v["auroc_ci"], v["auroc_not_deletion_like"]) for k, v in report["models"].items()}, indent=1))


if __name__ == "__main__":
    main(do_finetune="--no-finetune" not in sys.argv)
