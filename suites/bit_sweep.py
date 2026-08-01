"""Fragilitäts-Fläche: Bit-Tiefe × Checkpoint (Pythia-160m, RTN).

Vorregistrierte Erwartung: Schaden(Checkpoint)-Anstieg wird mit sinkender
Bit-Zahl steiler (Int8 flach — bereits gezeigt); der Checkpoint-Effekt
(143k/84k-Ratio) wächst monoton mit sinkender Bit-Tiefe.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_lockin import probe_texts, nll_on, rtn_quantize_, STEPS  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
MODEL = "EleutherAI/pythia-160m"
BITS = [8, 6, 5, 4, 3, 2]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = probe_texts()
    out = {}
    for s in STEPS:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=f"step{s}", dtype=torch.float32).to(dev).eval()
        basis = copy.deepcopy(model.state_dict())
        nll0 = nll_on(model, tok, texts, dev)
        row = {"fp32": round(nll0, 4)}
        for b in BITS:
            model.load_state_dict(basis)
            rtn_quantize_(model, b)
            row[f"delta_int{b}"] = round(nll_on(model, tok, texts, dev) - nll0, 4)
        out[s] = row
        del model, basis
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(s, row, flush=True)
    # Checkpoint-Effekt je Bit-Tiefe (143k vs 84k)
    ratio = {f"int{b}": round(out[143000][f"delta_int{b}"] /
                              (out[84000][f"delta_int{b}"] + 1e-9), 2) for b in BITS}
    res = {"flaeche": out, "ratio_143k_zu_84k_je_bit": ratio}
    (ART / "bit_sweep_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ratio, indent=1))


if __name__ == "__main__":
    main()
