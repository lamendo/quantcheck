"""KG-D0b: deklarierte einmalige Eskalation der Dekompression (Praereg Doc 33).

Aenderung ggue. D0 (dokumentierter Confound-Repair, KEIN Ergebnis-Shopping):
D0 scheiterte am ANKER, nicht am Regularisierer (auch lam=0.03 mit
unveraendertem Rang verlor 12% NLL -> Englisch-Wikitext-Drift gegen
DE-Probe). D0b nutzt die eigene Trainingsverteilung (Pile-Stichprobe),
lr 1e-5, 1000 Schritte, Arme lam in {0, 0.3, 1.0} (0 = Drift-Boden).
PASS-Kriterium UNVERAENDERT: eff_rang >= 3.0 UND nll <= Baseline*1.01.
Danach gilt das D0-Verdikt endgueltig.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_lockin import probe_texts  # noqa: E402
from decompress_kg0 import diff_effrank, measure, REPO, REV  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
LAMBDAS = [0.0, 0.3, 1.0]
STEPS, BS, ACCUM, SEQ, LR = 1000, 2, 4, 512, 1e-5


def train_blocks_pile(tok):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // SEQ) * SEQ
    blocks = ids[:n].view(-1, SEQ)
    g = torch.Generator().manual_seed(31)
    return blocks[torch.randperm(len(blocks), generator=g)]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(31)
    texts = probe_texts()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    blocks = train_blocks_pile(tok)
    print(f"{len(blocks)} Pile-Bloecke à {SEQ} Tokens", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, dtype=torch.float32).to(dev)
    baseline = measure(base, tok, texts, dev, "baseline")
    del base
    if dev == "cuda":
        torch.cuda.empty_cache()

    out = {"repo": REPO, "rev": REV, "steps": STEPS, "lr": LR,
           "anker": "NeelNanda/pile-10k",
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
            if step % 100 == 0:
                print(f"lam={lam} step={step} lm={o.loss.item():.3f} "
                      f"er={er.item():.2f}", flush=True)
        res = measure(m, tok, texts, dev, f"lam={lam}")
        res["pass_d0"] = bool(res["eff_rang"] >= 3.0 and
                              res["nll_fp32"] <= baseline["nll_fp32"] * 1.01)
        out["arme"][str(lam)] = res
        (ART / "decompress_kg0b_partial.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        del m, opt
        if dev == "cuda":
            torch.cuda.empty_cache()

    out["verdikt"] = ("KG-D0 PASS (via Eskalation D0b)"
                      if any(a["pass_d0"] for a in out["arme"].values())
                      else "KG-D0 ENDGUELTIG KILL")
    (ART / "decompress_kg0b_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
