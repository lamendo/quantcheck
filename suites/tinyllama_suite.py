"""Härtungstest Übertraining: TinyLlama-1.1B (3T Tokens, ~2700 Tok/Param).

VORREGISTRIERT (vor dem Lauf): Der Pythia-1b (300B Tokens) zeigte KEINE
Kompression und KEINEN Checkpoint-Effekt. Unser Mechanismus sagt voraus,
dass Übertraining beides erzeugt. Kriterien:
  K1  Spearman(Tokens, eff_rang)   ≤ −0.7  (progressive Kompression)
  K2  Spearman(Tokens, delta_int4) ≥ +0.7  (steigende Fragilität)
  K3  Spearman(eff_rang, delta_int4) ≤ −0.7 (Mechanismus-Link)
  BESTÄTIGT nur bei K1∧K2∧K3; sonst Kill → Befund bleibt Pythia-gebunden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_lockin import probe_texts, nll_on, rtn_quantize_, eff_rang  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
CKPTS = [  # (Repo, Tokens in Mrd.)
    ("TinyLlama/TinyLlama-1.1B-step-50K-105b", 105),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b", 503),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T", 1000),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T", 1500),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T", 2000),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", 2500),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 3000),
]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = probe_texts()
    part = ART / "tinyllama_partial.jsonl"
    rows = ([json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
            if part.exists() else [])
    done = {r["tokens_b"] for r in rows}
    for repo, tokens in CKPTS:
        if tokens in done:
            print(f"skip {tokens}b (bereits gemessen)", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        base = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(dev).eval()
        nll0 = nll_on(base, tok, texts, dev)
        rang = eff_rang(base, tok, texts, dev)
        del base
        if dev == "cuda":
            torch.cuda.empty_cache()
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(dev).eval()
        rtn_quantize_(m, 4)
        d4 = nll_on(m, tok, texts, dev) - nll0
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        rows.append({"tokens_b": tokens, "nll_fp32": round(nll0, 4),
                     "eff_rang": round(rang, 2), "delta_int4": round(d4, 4)})
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        print(rows[-1], flush=True)
    rows.sort(key=lambda r: r["tokens_b"])
    t = [r["tokens_b"] for r in rows]
    rg = [r["eff_rang"] for r in rows]
    d4 = [r["delta_int4"] for r in rows]
    k1, k2, k3 = spearman(t, rg), spearman(t, d4), spearman(rg, d4)
    out = {"checkpoints": rows,
           "K1_tokens_rang": round(k1, 3), "K2_tokens_schaden": round(k2, 3),
           "K3_rang_schaden": round(k3, 3),
           "verdikt": ("ÜBERTRAINING-THESE BESTÄTIGT"
                       if k1 <= -0.7 and k2 >= 0.7 and k3 <= -0.7
                       else "KILL: Befund bleibt (vorerst) Pythia-gebunden")}
    (ART / "tinyllama_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
