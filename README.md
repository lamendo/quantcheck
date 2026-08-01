# quantcheck — pick the right checkpoint before you quantize

**TL;DR:** Late in LM pretraining, models keep the same benchmark scores but
become dramatically more fragile to 4-bit quantization. We trace this to a
measurable internal event (depth-schedule rank compression after a "lock-in"
transition) and provide a cheap, label-free probe that tells you **which
checkpoint to quantize** — up to ~4× less Q4 damage at equal quality.

![Pythia-160m: rank compresses after lock-in, Int4 damage follows](figures/pythia160m_lockin.png)

![Three families, three regimes](figures/three_families.png)

Longer read: [WRITEUP.md](WRITEUP.md) · interactive explainer:
[explainer/](explainer/) · full pre-registration ledger incl. negatives:
[PREREG.md](PREREG.md) · original lab notes (German): [docs/research_trail/](docs/research_trail/)

## The finding (Pythia suite)

| Pythia-160m checkpoint | PPL f16 | PPL Q4_K_M | ΔlnPPL (damage) |
|---|---|---|---|
| step 84 000 | 1.655 | 1.716 | **0.036** |
| step 143 000 (final) | 1.679 | 1.937 | **0.142 (3.9×)** |

- Damage onset coincides with an endogenous training event ("lock-in") that
  has no counterpart in the LR schedule and is invisible to loss/benchmarks.
- Mechanism candidate: post-lock-in **rank compression** of the mean depth
  update ("schedule"); Spearman(rank, Int4 damage) = −0.96 (160m) / −0.86 (410m).
- Replicated: **4 model scales (70m/160m/410m/1b)** × full quant spectrum —
  RTN 2–8 bit AND production GGUF Q2_K…Q8_0 (checkpoint-damage ratio
  2.4–10.5× across the menu, amplifying toward low bits: 13× at 2-bit RTN)
  × two probe languages (DE/EN). Rank↔damage chain: 70m (1.13 → **8.15**)
  · 160m (1.62 → 3.71) · 1b (7.02 → 0.55) · 410m (8.05 → 1.11). The 1b
  barely compresses at 300B tokens and barely suffers — **the effect hits
  small models hardest, exactly the ones quantized most aggressively.**
- **Second family (modern recipe): OLMo-2-1B, 2025.** 9 checkpoints over
  4T tokens: rank compresses continuously 9.7 → 3.7, Int4 damage rises
  0.27 → 0.76 peak (2.8×), rank↔damage ρ = −0.87. Milder than Pythia's
  cliff, same direction, same probe.

## Usage

```bash
# 1. Rank curve over checkpoints (the telemetry)
python rank_probe.py --hf-model EleutherAI/pythia-160m --steps 64000:143000:8000

# 2. Quantization damage curve (RTN proxy, fast)
python quant_probe.py --hf-model EleutherAI/pythia-160m --probe en

# 3. Recommendation: earliest checkpoint after benchmark maturity,
#    before rank compression exceeds your damage budget.
```

## What this is / isn't

- ✅ A label-free, forward-pass-only probe (no benchmarks, no training).
- ✅ Reproducible: all scripts, all numbers, all pre-registrations included.
- ✅ **Replicated on a modern 2025 recipe:** OLMo-2-1B (9 checkpoints,
  84B–4T tokens): all three pre-registered criteria hit (ρ tokens↔rank
  −0.88, tokens↔damage +0.87, rank↔damage −0.87; final/min damage 2.8×).
  Profile differs from Pythia — continuous compression instead of a late
  lock-in event — but the rank↔damage link holds.
- ✅ **Includes a pre-registered negative replication:** TinyLlama-1.1B
  (3T tokens, ~2,700 tok/param) shows NO compression and NO Int4 fragility.
  Three families, three regimes (late cliff / continuous drift / immune) —
  which is exactly why you should *measure* your suite instead of assuming.
- ❌ Not a claim about reasoning, capability, or model quality per se.
- ❌ Not a universal law: it's a measurable regime. Confirmed on Pythia
  (4 scales) + OLMo-2-1B, refuted on TinyLlama; **community replication on
  other suites (Amber, SmolLM, OLMo-7B…) is the explicit ask.**

## Repo layout

```
quantcheck/
  rank_probe.py          # eff. rank of mean depth-update over checkpoints
  quant_probe.py         # fp32 vs RTN-Int4/Int8 ΔNLL per checkpoint (probe corpora included)
  gguf_probe.py          # real GGUF Q4_K_M via llama.cpp (optional)
  suites/                # full checkpoint-suite runs: olmo, tinyllama, olmoe (MoE),
                         # bit sweep, gguf spectrum, and the pre-registered
                         # (failed) post-hoc decompression experiment
  results/               # all JSONs incl. pre-registered criteria & verdicts
  explainer/             # interactive visual explainer (self-contained HTML)
  PREREG.md              # ledger: what was predicted before each run, incl. kills
  WRITEUP.md             # the long-form writeup (blog draft)
  figures/               # core result figures (generated from results/)
  docs/research_trail/   # original lab notes (German), as-run
  patches/               # small env fixes needed for reproduction
  CITATION.cff           # how to cite
  requirements.txt
  ANNOUNCEMENTS.md       # release notes drafts
```

Note: code comments are partly in German — these are the actual research
scripts, published as-run. Cleanup PRs welcome; numbers won't change.

## Provenance & method

Built with a falsification-first workflow (pre-registered endpoints,
kill criteria, negative results published). Full research trail:
lamendo research — "Representation Observatory" series. License: Apache-2.0.
