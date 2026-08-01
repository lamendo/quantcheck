"""Quantisierungs-Robustheit × Lock-in — zählt die späte Reorganisation für Q4?

Design (vorregistriert):
  Checkpoints Pythia-160m: 64k · 76k · 84k(Lock) · 92k · 108k · 128k · 143k
  je Checkpoint: NLL auf Sondenkorpus (31 Texte) in fp32, dann Gewichte
  RTN-quantisiert (Int4 per-Output-Kanal, symmetrisch; alle nn.Linear),
  NLL erneut. Schaden = ΔNLL = NLL_q − NLL_fp32. Int8 = Kontrolle (≈0 erwartet).
  VERDIKT „Lock-in-relevant": mittlerer Int4-Schaden PRE (64k–84k) vs POST
  (92k–143k) unterscheidet sich um ≥20 % relativ UND der Verlauf verschiebt
  sich am Fenster monoton. Sonst: negativ (Schaden lock-in-unabhängig).
  Deklaration: Weight-RTN ist ein Proxy (GGUF-K-Quants sind feiner);
  Aktivierungs-Quantisierung ist NICHT abgedeckt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ART = HERE.parent / "artifacts" / "topo_steering"
MODEL = "EleutherAI/pythia-160m"
STEPS = [64000, 76000, 84000, 92000, 108000, 128000, 143000]


def probe_texts():
    import qwythos_extract as qx
    catalog = {i["id"]: i for i in json.loads(
        (HERE / "catalog.json").read_text(encoding="utf-8"))}
    return [catalog[i]["prompt"] for i in qx.PICK] + [q for q, _ in qx.REASONING]


PROBE_EN = [
    "The train leaves the station at 9:20 in the morning and the journey takes 85 minutes. At what time does it arrive?",
    "Anna has 24 apples. She buys 13 more and gives 9 away. Then her stock doubles. How many apples does she have now?",
    "Compute ((17 + 26) * 3 - 15) / 2 and explain each step of the calculation carefully.",
    "Peter is taller than Susan. Susan is taller than Tom. Tom is taller than Maria. Who is the shortest of the four?",
    "Given the list 7, 19, 3, 25, 11, 8, 31, 2, how many of these numbers are greater than 10?",
    "A farmer has 45 sheep. He sells a third of them and then buys 12 new ones. How many sheep does he own afterwards?",
    "The capital of France is Paris, and the capital of Italy is Rome. Which river flows through the German capital?",
    "Write a short Python function that returns the maximum of two numbers and briefly explain how it works.",
    "A library opens at 8:45 and closes 9 hours and 30 minutes later. At what time does it close in the evening?",
    "Lisa reads 12 pages every day. How many pages has she read after two weeks, and how long does she need for a 300-page book?",
    "The old lighthouse stood on the cliff for two hundred years before the storm finally broke its lantern room.",
    "In the quiet valley below the mountains, the village bakery opened its doors long before the sun had risen.",
    "Scientists measured the temperature of the lake every morning and noticed a slow but steady increase over the years.",
    "The museum's new exhibition explains how early printing presses changed the way knowledge spread through Europe.",
    "After the long drought, the first heavy rain filled the reservoirs and the farmers finally planted their fields again.",
    "The committee discussed the proposal for three hours but could not agree on the budget for the coming year.",
    "A small robot moved carefully along the warehouse shelves, scanning each box and reporting its position to the server.",
    "The recipe requires 250 grams of flour, two eggs, and half a liter of milk, mixed slowly until the batter is smooth.",
    "During the night shift, the engineer checked the turbine readings twice and wrote a short report for the morning crew.",
    "The children built a small dam of stones in the stream and watched how the water found a new path around it.",
]


def probe_texts_en():
    return PROBE_EN * 2


def nll_on(model, tok, texts, dev):
    tot, n = 0.0, 0
    for tx in texts:
        ids = tok(tx, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**ids)
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        nxt = ids["input_ids"][0, 1:]
        tot += float(-lp[:-1].gather(1, nxt.unsqueeze(1)).sum())
        n += int(nxt.shape[0])
    return tot / n


def rtn_quantize_(model, bits):
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight.data
                scale = W.abs().amax(dim=1, keepdim=True) / qmax
                scale = torch.clamp(scale, min=1e-12)
                mod.weight.data = torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def eff_rang(model, tok, texts, dev, n_texts=12):
    sums, cnt = None, 0
    for tx in texts[:n_texts]:
        ids = tok(tx, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        h = torch.stack([x[0] for x in out.hidden_states]).float().cpu().numpy()
        d = np.diff(h[:, 1:, :], axis=0)
        s = d.sum(axis=1)
        sums = s if sums is None else sums + s
        cnt += d.shape[1]
    mu = sums / cnt
    sv = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
    return float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))


def main():
    import argparse
    import sys
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default=MODEL)
    ap.add_argument("--out", default="quant_lockin_results.json")
    ap.add_argument("--probe", choices=["de", "en"], default="de")
    ap.add_argument("--steps", default="", help="Komma-Liste, überschreibt Default")
    args = ap.parse_args()
    global STEPS
    if args.steps:
        STEPS = [int(x) for x in args.steps.split(",")]
    hf_model = args.hf_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hf_model)
    texts = probe_texts() if args.probe == "de" else probe_texts_en()
    rows = {}
    for s in STEPS:
        rev = f"step{s}"
        base = AutoModelForCausalLM.from_pretrained(
            hf_model, revision=rev, dtype=torch.float32).to(dev).eval()
        nll0 = nll_on(base, tok, texts, dev)
        vals = {"nll_fp32": round(nll0, 4),
                "eff_rang": round(eff_rang(base, tok, texts, dev), 2)}
        del base                       # VOR den Quant-Kopien freigeben (8-GB-GPU!)
        if dev == "cuda":
            torch.cuda.empty_cache()
        for bits, key in ((8, "int8"), (4, "int4")):
            m = AutoModelForCausalLM.from_pretrained(
                hf_model, revision=rev, dtype=torch.float32).to(dev).eval()
            rtn_quantize_(m, bits)
            nq = nll_on(m, tok, texts, dev)
            vals[f"nll_{key}"] = round(nq, 4)
            vals[f"delta_{key}"] = round(nq - nll0, 4)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        rows[s] = vals
        print(s, vals, flush=True)

    if not all(k in rows for k in (64000, 76000, 84000, 92000, 108000, 128000, 143000)):
        (ART / args.out).write_text(json.dumps({"checkpoints": rows}, indent=1),
                                    encoding="utf-8")
        print(json.dumps(rows, indent=1))
        return
    pre = [rows[s]["delta_int4"] for s in (64000, 76000, 84000)]
    post = [rows[s]["delta_int4"] for s in (92000, 108000, 128000, 143000)]
    mp, mq = float(np.mean(pre)), float(np.mean(post))
    rel = (mq - mp) / (abs(mp) + 1e-9)
    d4 = [rows[s]["delta_int4"] for s in STEPS]
    monotone_shift = (np.argsort(d4)[:3].tolist() in
                      ([0, 1, 2], [2, 1, 0]) or np.argsort(d4)[-3:].tolist()
                      in ([4, 5, 6], [6, 5, 4]))
    out = {"checkpoints": rows,
           "int4_schaden_pre_mean": round(mp, 4),
           "int4_schaden_post_mean": round(mq, 4),
           "relative_aenderung": round(rel, 3),
           "verdikt": ("LOCK-IN-RELEVANT" if abs(rel) >= 0.2 and monotone_shift
                       else "NEGATIV: Q4-Schaden lock-in-unabhängig")}
    (ART / args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
