"""GGUF-Härtung: echter Q4_K_M-Schaden bei Checkpoint 84k vs 143k (Pythia-160m).

Vorregistriert: RTN-Befund bestätigt, wenn ΔlnPPL(Q4_K_M vs f16) bei 143k
mindestens 2× so groß ist wie bei 84k. Sondenkorpus deutsch (deklariert,
konsistent zum RTN-Test).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
LLAMA = HERE.parent / "_tools" / "llama.cpp"
BIN = LLAMA / "build" / "bin"
WORK = Path(r"D:\mentis_artifacts\gguf_quant")
ART = HERE.parent / "artifacts" / "topo_steering"
STEPS = [84000, 143000]
PY = r"C:\mentis_ai\.venv_sense\Scripts\python.exe"


def probe_file():
    import sys
    sys.path.insert(0, str(HERE))
    import qwythos_extract as qx
    catalog = {i["id"]: i for i in json.loads(
        (HERE / "catalog.json").read_text(encoding="utf-8"))}
    texts = [catalog[i]["prompt"] for i in qx.PICK] + [q for q, _ in qx.REASONING]
    cot = json.loads((HERE / "catalog_cot.json").read_text(encoding="utf-8"))
    texts += [c["prompt"] for c in cot]
    p = WORK / "probe_de.txt"
    p.write_text("\n\n".join(texts * 2), encoding="utf-8")
    return p


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]}: rc={r.returncode}\n{r.stderr[-1500:]}")
    return r.stdout + r.stderr


def ppl_of(gguf, probe):
    out = run([str(BIN / "llama-perplexity"), "-m", str(gguf), "-f", str(probe),
               "-c", "512", "-t", "8"])
    m = re.findall(r"Final estimate: PPL = ([0-9.]+)", out)
    if not m:
        m = re.findall(r"PPL = ([0-9.]+)", out)
    return float(m[-1])


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    probe = probe_file()
    from huggingface_hub import snapshot_download
    out = {}
    import math
    for s in STEPS:
        snap = snapshot_download("EleutherAI/pythia-160m", revision=f"step{s}")
        f16 = WORK / f"pythia160m_step{s}_f16.gguf"
        q4 = WORK / f"pythia160m_step{s}_q4km.gguf"
        if not f16.exists():
            run([PY, str(LLAMA / "convert_hf_to_gguf.py"), snap,
                 "--outfile", str(f16), "--outtype", "f16"])
        if not q4.exists():
            run([str(BIN / "llama-quantize"), str(f16), str(q4), "q4_k_m"])
        p16 = ppl_of(f16, probe)
        p4 = ppl_of(q4, probe)
        out[s] = {"ppl_f16": round(p16, 3), "ppl_q4km": round(p4, 3),
                  "delta_lnppl": round(math.log(p4) - math.log(p16), 4)}
        print(s, out[s], flush=True)
    ratio = out[143000]["delta_lnppl"] / (out[84000]["delta_lnppl"] + 1e-9)
    out["ratio_143k_zu_84k"] = round(ratio, 2)
    out["verdikt"] = ("RTN-BEFUND BESTÄTIGT (GGUF-K-Quant)" if ratio >= 2.0
                      else "NICHT bestätigt — K-Quants mitigieren den Effekt")
    (ART / "gguf_quant_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
