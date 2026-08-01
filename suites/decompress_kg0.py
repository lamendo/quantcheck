"""KG-D0 Moonshot Dekompression: Ist der Fahrplan-Rang billig anhebbar?

Präregistrierung: docs/research/representation_observatory/33_MOONSHOT_DEKOMPRESSION.md
PASS: ein λ aus {0.03, 0.1, 0.3} erreicht eff_rang >= 3.0 UND nll <= Baseline*1.01.
delta_int4 wird deskriptiv mitgemessen, entscheidet aber erst KG-D1.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_lockin import probe_texts, nll_on, rtn_quantize_, eff_rang  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
REPO = "EleutherAI/pythia-160m"
REV = "step143000"
LAMBDAS = [0.03, 0.1, 0.3]
STEPS, BS, ACCUM, SEQ, LR = 300, 2, 4, 512, 2e-5


def train_blocks(tok):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // SEQ) * SEQ
    blocks = ids[:n].view(-1, SEQ)
    g = torch.Generator().manual_seed(31)
    perm = torch.randperm(len(blocks), generator=g)
    return blocks[perm]


def diff_effrank(hidden_states):
    """Differenzierbarer effektiver Rang des Batch-Fahrplans (Serie-Definition)."""
    h = torch.stack(hidden_states)          # (L+1, B, T, d)
    mu = (h[1:] - h[:-1])[:, :, 1:, :].mean(dim=(1, 2)).float()  # (L, d)
    s = torch.linalg.svdvals(mu - mu.mean(0))
    return (s.sum() ** 2) / (s.pow(2).sum() + 1e-12)


def measure(model, tok, texts, dev, tag):
    model.eval()
    nll = nll_on(model, tok, texts, dev)
    rang = eff_rang(model, tok, texts, dev)
    q = copy.deepcopy(model)
    rtn_quantize_(q, 4)
    d4 = nll_on(q, tok, texts, dev) - nll
    del q
    if dev == "cuda":
        torch.cuda.empty_cache()
    r = {"nll_fp32": round(nll, 4), "eff_rang": round(rang, 2),
         "delta_int4": round(d4, 4)}
    print(tag, r, flush=True)
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(31)
    texts = probe_texts()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    blocks = train_blocks(tok)
    print(f"{len(blocks)} Trainingsbloecke à {SEQ} Tokens", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, dtype=torch.float32).to(dev)
    baseline = measure(base, tok, texts, dev, "baseline")
    del base
    if dev == "cuda":
        torch.cuda.empty_cache()

    out = {"repo": REPO, "rev": REV, "steps": STEPS,
           "tokens_trainiert": STEPS * BS * ACCUM * SEQ,
           "baseline": baseline, "arme": {}}
    for lam in LAMBDAS:
        torch.manual_seed(31)
        m = AutoModelForCausalLM.from_pretrained(
            REPO, revision=REV, dtype=torch.float32).to(dev)
        m.gpt_neox.embed_in.requires_grad_(False)
        m.embed_out.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr=LR)
        m.train()
        bi = 0
        for step in range(STEPS):
            opt.zero_grad()
            for _ in range(ACCUM):
                x = blocks[bi % len(blocks)].unsqueeze(0).repeat(BS, 1).to(dev)
                bi += 1
                o = m(input_ids=x, labels=x, output_hidden_states=True)
                er = diff_effrank(o.hidden_states)
                loss = (o.loss - lam * er) / ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"lam={lam} step={step} lm={o.loss.item():.3f} "
                      f"er={er.item():.2f}", flush=True)
        res = measure(m, tok, texts, dev, f"lam={lam}")
        res["pass_d0"] = bool(res["eff_rang"] >= 3.0 and
                              res["nll_fp32"] <= baseline["nll_fp32"] * 1.01)
        out["arme"][str(lam)] = res
        del m, opt
        if dev == "cuda":
            torch.cuda.empty_cache()

    out["verdikt"] = ("KG-D0 PASS" if any(a["pass_d0"] for a in out["arme"].values())
                      else "KG-D0 KILL (Eskalation 1000 Schritte / lam=1.0 erlaubt, dann final)")
    (ART / "decompress_kg0_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
