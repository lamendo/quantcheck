# Reproduce every number

One command per published result. All scripts write JSON into `results/`
(the shipped files are our original outputs — rerun and diff).

## Environment (as run)

- Windows 11 Pro · Python 3.11.9 · CUDA 12.1 · GPU: RTX 3070 8 GB
  (everything also runs CPU-only, slower; the OLMoE suite ran CPU/bf16)
- Exact package versions: `requirements-lock.txt` (floors in `requirements.txt`;
  note: scripts use the transformers ≥5 `dtype=` kwarg)
- llama.cpp (GGUF results only): upstream commit `0badc06` (2026-07-10),
  CMake default release build, targets `llama-quantize`, `llama-perplexity`;
  plus the one-line conversion fix in `patches/gptneox_transformers5.md`
- HF checkpoints are addressed via revision names (`step84000`,
  `stage1-step…`, branch names of the official repos). These branches are
  de-facto frozen by their publishers, but we did not pin commit SHAs —
  a re-run fetches whatever those refs point to.

## Core results

| Published result | Command |
|---|---|
| `results/quant_lockin_results.json` (160m lock-in × Int4/Int8, DE probe) | `python quant_probe.py` |
| `results/quant_lockin_160m_en.json` (EN probe) | `python quant_probe.py --probe en --out quant_lockin_160m_en.json` |
| `results/quant_lockin_{410m,70m,1b}.json` (scale chain) | `python quant_probe.py --hf-model EleutherAI/pythia-410m --steps 64000,84000,108000,143000 --out quant_lockin_410m.json` (analog 70m/1b; 1b used the cached-step subset in the JSON) |
| `results/pythia_emergence.json` (rank/rogue/form over training) | `python rank_probe.py` |
| `results/pythia_fein.json` (fine sweep 64k–128k) | `python rank_probe.py --steps 64000:128000:4000 --out pythia_fein.json` |
| `results/bit_sweep_results.json` (RTN 2–8 bit, 160m) | `python suites/bit_sweep.py` |
| `results/gguf_quant_results.json` (real Q4_K_M, 160m 84k vs 143k) | `QUANTCHECK_LLAMACPP=/path/to/llama.cpp python gguf_probe.py` |
| `results/gguf_spectrum_results.json` (Q2_K…Q8_0, 160m) | same env, `python suites/gguf_spectrum.py` (needs the f16 GGUFs produced by gguf_probe first) |
| `results/olmo_results.json` (OLMo-2-1B, 9 ckpts) | `python suites/olmo_suite.py` |
| `results/tinyllama_results.json` (TinyLlama, 7 ckpts) | `python suites/tinyllama_suite.py` |
| `results/olmoe_partial_4of8.json` (MoE, suspended) | `python suites/olmoe_suite.py` (resumes from the shipped partial; ~14 GB commit memory per load) |
| `results/pythia_benchmark_maturity.json` (the "benchmarks flat" side) | `python suites/benchmark_maturity.py` (fetches EleutherAI's published per-checkpoint evals from github.com/EleutherAI/pythia) |
| `results/decompress_kg0*.json` (failed post-hoc repair) | `python suites/decompress_kg0.py`, then `python suites/decompress_kg0b.py` |
| `figures/*.png` | regenerated from `results/` (plot code trivial; open an issue if you want it shipped) |

## Known reproduction caveats (honest list)

1. **EN probe file history:** our original EN runs used the 20 texts listed
   twice (40 entries). This provably does not change any number — NLL is a
   per-token mean (duplication cancels exactly) and the rank probe uses only
   the first 12 texts — but the shipped `probes/probe_en.json` now contains
   the 20 unique texts. Re-runs reproduce the published EN values.
2. **GGUF toolchain:** PPL parsing depends on llama-perplexity's output
   format at the pinned commit; newer llama.cpp may need a regex touch.
3. **Effective rank across model sizes:** raw values depend on depth L
   (upper bound = number of layer transitions). Within-family curves are the
   claim; the cross-scale table is directional and should be read with the
   normalized rank (r_eff / min(L−1, hidden)) in mind.
4. **No externally timestamped preregistration:** PREREG.md is our internal
   ledger, written before each run as part of the workflow, but the public
   git history begins 2026-08-01. Treat it as an as-written lab record, not
   a third-party-verified preregistration.
5. Spearman values in the shipped JSONs were computed with a tie-naive rank
   correlation (double argsort). Our series contain no ties, so
   `scipy.stats.spearmanr` (now the default in the suites) yields identical
   values; re-runs use scipy.
