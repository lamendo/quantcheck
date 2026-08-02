"""Robustness check for the effective-rank probe: bootstrap over probe texts.

Addresses the fair criticism that the probe corpus is small (31 texts) and
the rank is computed from the first 12: we compute per-text depth-update
contributions once, then resample 12-text subsets 2000x and report the
full-corpus value plus a bootstrap 95% interval for the two headline
checkpoints (84k = robust regime, 143k = fragile regime, Pythia-160m).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

MODEL = "EleutherAI/pythia-160m"
STEPS = [84000, 143000]
N_BOOT, SUBSET = 2000, 12


def per_text_contribs(model, tok, texts, dev):
    out = []
    for tx in texts:
        ids = tok(tx, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        h = torch.stack([x[0] for x in o.hidden_states]).float().cpu().numpy()
        d = np.diff(h[:, 1:, :], axis=0)
        out.append((d.sum(axis=1), d.shape[1]))          # (D-1,H), n_tokens
    return out


def rang_of(contribs, idx):
    s = sum(contribs[i][0] for i in idx)
    c = sum(contribs[i][1] for i in idx)
    mu = s / c
    sv = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
    return float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = json.loads((HERE / "probes" / "probe_de.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(31)
    res = {}
    for s in STEPS:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=f"step{s}", dtype=torch.float32).to(dev).eval()
        contribs = per_text_contribs(m, tok, texts, dev)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        full = rang_of(contribs, range(len(texts)))
        first12 = rang_of(contribs, range(SUBSET))       # the published estimator
        boots = [rang_of(contribs, rng.choice(len(texts), SUBSET, replace=False))
                 for _ in range(N_BOOT)]
        res[str(s)] = {
            "eff_rang_first12_published_estimator": round(first12, 3),
            "eff_rang_all31": round(full, 3),
            "bootstrap_median": round(float(np.median(boots)), 3),
            "bootstrap_ci95": [round(float(np.percentile(boots, 2.5)), 3),
                               round(float(np.percentile(boots, 97.5)), 3)],
            "n_boot": N_BOOT, "subset_size": SUBSET,
        }
        print(s, res[str(s)], flush=True)
    sep = (res[str(STEPS[0])]["bootstrap_ci95"][0] >
           res[str(STEPS[1])]["bootstrap_ci95"][1])
    res["intervals_disjoint_84k_vs_143k"] = bool(sep)
    (HERE / "results" / "rank_bootstrap.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
