"""GGUF-Spektrum: Q2_K…Q8_0 auf Checkpoint 84k vs 143k (aus vorhandenen f16-GGUFs)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from gguf_probe import WORK, ART, BIN, probe_file, run, ppl_of  # noqa: E402

QUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
STEPS = [84000, 143000]


def main():
    probe = probe_file()
    out = {}
    for s in STEPS:
        f16 = WORK / f"pythia160m_step{s}_f16.gguf"
        p16 = ppl_of(f16, probe)
        row = {"ppl_f16": round(p16, 3)}
        for q in QUANTS:
            qf = WORK / f"pythia160m_step{s}_{q}.gguf"
            if not qf.exists():
                run([str(BIN / "llama-quantize"), str(f16), str(qf), q])
            pq = ppl_of(qf, probe)
            row[q] = round(math.log(pq) - math.log(p16), 4)
        out[s] = row
        print(s, row, flush=True)
    out["ratio_143k_zu_84k"] = {q: round(out[143000][q] / (out[84000][q] + 1e-9), 2)
                                for q in QUANTS}
    (ART / "gguf_spectrum_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["ratio_143k_zu_84k"], indent=1))


if __name__ == "__main__":
    main()
