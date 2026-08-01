"""Studie 32: OLMo-2-1B-Checkpoint-Sweep — der Entscheidungstest der Publikation.

VORREGISTRIERT (vor dem Lauf): Der Pythia-Befund (späte Rang-Kompression →
Int4-Fragilität) ist bisher auf EINER Familie belegt; TinyLlama (moderne
Rezeptur, 3T Tokens) war negativ. OLMo-2-0425-1B (AI2, 2025, modernste
öffentliche Rezeptur, 195 Stage-1-Checkpoints 0–4T Tokens) ist die dritte
Familie und entscheidet:

  BESTÄTIGT (K1∧K2∧K3) → Phänomen existiert in moderner Rezeptur;
    Publikation mit „2 positive + 1 negative Familie — miss dein Regime".
  KILL → Pythia-Kuriosum; große Publikation ABGESAGT, höchstens Methoden-Notiz.

Kriterien (identisch TinyLlama-Härtung, Vergleichbarkeit):
  K1  Spearman(Tokens, eff_rang)   <= -0.7  (progressive Kompression)
  K2  Spearman(Tokens, delta_int4) >= +0.7  (steigende Fragilität)
  K3  Spearman(eff_rang, delta_int4) <= -0.7 (Mechanismus-Link)
Zusatz-Deskriptiv (kein Kriterium): Verhältnis delta_int4 final / minimum
(Pythia-160m: ~4x) als Effektstärke fürs Writeup.

10 Checkpoints: Früh-Anker 84B, dann 441B..4001B gleichmäßig. step0
ausgeschlossen (untrainiert, keine Aussage über Trainingsdynamik).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from quant_probe import probe_texts, nll_on, rtn_quantize_, eff_rang  # noqa: E402

ART = HERE.parent / "results"
REPO = "allenai/OLMo-2-0425-1B"
CKPTS = [  # (Revision, Tokens in Mrd.)
    ("stage1-step40000-tokens84B", 84),
    ("stage1-step210000-tokens441B", 441),
    ("stage1-step450000-tokens944B", 944),
    ("stage1-step700000-tokens1469B", 1469),
    ("stage1-step940000-tokens1972B", 1972),
    ("stage1-step1180000-tokens2475B", 2475),
    ("stage1-step1430000-tokens2999B", 2999),
    ("stage1-step1670000-tokens3503B", 3503),
    ("stage1-step1907359-tokens4001B", 4001),
]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = probe_texts()
    part = ART / "olmo_partial.jsonl"
    rows = ([json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
            if part.exists() else [])
    done = {r["tokens_b"] for r in rows}
    for rev, tokens in CKPTS:
        if tokens in done:
            print(f"skip {tokens}B (bereits gemessen)", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(REPO, revision=rev)
        base = AutoModelForCausalLM.from_pretrained(
            REPO, revision=rev, dtype=torch.float32).to(dev).eval()
        nll0 = nll_on(base, tok, texts, dev)
        rang = eff_rang(base, tok, texts, dev)
        del base
        if dev == "cuda":
            torch.cuda.empty_cache()
        m = AutoModelForCausalLM.from_pretrained(
            REPO, revision=rev, dtype=torch.float32).to(dev).eval()
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
    ratio = (max(d4) / max(min(d4), 1e-6)) if min(d4) > 0 else None
    out = {"repo": REPO, "checkpoints": rows,
           "K1_tokens_rang": round(k1, 3), "K2_tokens_schaden": round(k2, 3),
           "K3_rang_schaden": round(k3, 3),
           "effektstaerke_final_zu_min": round(ratio, 2) if ratio else None,
           "verdikt": ("BESTÄTIGT: Phänomen existiert in moderner Rezeptur"
                       if k1 <= -0.7 and k2 >= 0.7 and k3 <= -0.7
                       else "KILL: Pythia-Kuriosum — große Publikation absagen")}
    (ART / "olmo_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
