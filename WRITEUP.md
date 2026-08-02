# An observation about late pretraining and 4-bit fragility — and an open call for measurements

**Status: v1 (2026-08-02).** This is an exploratory write-up of what we
could measure on one consumer GPU — not a finished finding. We lack the
resources for more families, larger models and modern calibration quants.
Everything needed to check or extend this is in the repo; one command
suffices. Please measure and report.

---

**TL;DR.** On the checkpoint suites we could afford to measure, we observed:
on Pythia, zero-shot benchmarks saturate early (~step 13k of 143k), but the
*final* checkpoint is the worst one to quantize — Q4_K_M damage at step 143k
is **3.9x** the damage at step 84k at essentially equal fp16 quality, and the
onset coincides with a measurable internal transition ("lock-in") followed by
rank compression of the mean depth-update. The pattern replicates on OLMo-2-1B
(different shape) and is **absent** on TinyLlama — a regime, not a law, and
correlational, not proven causal. The forward-pass probe that measures it
runs on any HF checkpoint suite.

### 1 · Setup
Pythia-70m/160m/410m/1b, public checkpoints. Probe corpus: fixed 31-prompt
set (DE; EN replication with 20 prompts on 160m). Damage metric: ΔNLL
(fp32 vs RTN-Int4; Int8 as null control ≈ 0 everywhere) and ΔlnPPL
(f16-GGUF vs Q4_K_M via llama.cpp) — both pre-registered.

### 2 · Results
- Benchmark maturity (90 % of final): step ~13k (160m; public eval data).
- Lock-in event at ~84k: schedule-form snaps to its cross-family attractor,
  rank compression starts (5.4 → 1.6 by 143k), rogue-dim inventory reshuffles.
  No LR-schedule counterpart (cosine is featureless there).
- Int4 damage: 0.55 → 3.71 ΔNLL from 64k → 143k (160m); GGUF Q4_K_M
  confirms 3.9× (84k vs 143k). Spearman(rank, damage) = −0.96.
- **Whole-spectrum check (RTN 2–8 bit):** the checkpoint effect exists at
  every usable bit depth and AMPLIFIES as bits shrink — final/sweet-spot
  damage ratio ≈5× at 6 bit, 4.8× at 4 bit, **13× at 2 bit**; Int8 flat
  (null control).
- **Production GGUF spectrum (Q2_K…Q8_0, llama.cpp):** ratio 143k/84k =
  2.4× (Q8_0, tiny absolute) · 3.4× (Q6_K) · 10.5× (Q5_K_M) · 3.9× (Q4_K_M)
  · 4.4× (Q3_K_M) · 4.2× (Q2_K, absolute 0.63→2.67). The rule holds across
  the entire real-world quant menu.
- Scale chain (prediction-tested): **70m** compresses deepest (rank 1.13)
  and is most fragile (final damage **8.15**) · 160m (1.62 → 3.71) ·
  **1b (7.02 → 0.55)** · 410m (8.05 → 1.11). Strong negative rank↔damage
  relation across four scales (not perfectly monotone at the top pair —
  1b/410m sit close in rank; layer-count confound declared, L=6…24).
- **Scale-law twist (new):** the 1b shows essentially NO late compression
  within Pythia's 300B-token budget (rank flat 6.1→7.0) — and accordingly
  almost no checkpoint effect (pre 0.29 → post 0.50). The late fragility
  regime appears to be a **small-model phenomenon** at fixed token budget —
  i.e. checkpoint choice matters most exactly for the models people
  quantize hardest.
- EN-probe replication (160m): same curve shape as DE, damage
  0.34 → 2.94 (+283 % pre/post) — not a corpus artifact.
- Footnote: post-hoc 1-bit is trivially destructive (see 2-bit numbers);
  BitNet-style 1.58-bit is *training-time* quantization — our findings
  suggest one reason QAT works: it never enters the late fragility regime.

### 3 · Working rule (on the suites where the regime appears)
If you have intermediate checkpoints and deploy 4-bit: **prefer the earliest
checkpoint after benchmark maturity, before rank compression exceeds your
damage budget.** On Pythia-160m that window is roughly steps 80–95k — same
benchmarks, ~4x less Q4 damage than final. In TinyLlama-like regimes the rule
is a no-op (nothing to avoid). Whether YOUR run has such a window is exactly
what the probe measures — we cannot know it for you.

### 4 · Three families, three regimes
Pre-registered replications on two further checkpoint suites:
- **OLMo-2-1B (2025 recipe, 9 ckpts / 4T tokens): CONFIRMED** — rank
  compresses continuously 9.7 → 3.7, Int4 damage 0.27 → 0.76 peak (2.8×),
  all three criteria hit (tokens↔rank −0.88, tokens↔damage +0.87,
  rank↔damage −0.87). Different *shape* than Pythia (continuous drift, no
  late lock-in event) — same mechanism link, same practical rule.
- **TinyLlama-1.1B (3T tokens, ~2,700 tok/param, 7 ckpts): NEGATIVE** —
  no compression, damage ≈ 0 from 1T–3T; all three criteria missed.

So this is **not a universal law — it's a measurable regime** (late cliff /
continuous drift / immune), and the rank probe tells you which one your
suite is in *before* you quantize.

### 4b · Honest limits & call for replication
RTN + GGUF quant menu (AWQ/GPTQ untested); probe corpora small and declared;
correlation ≠ proven causation (compression → fragility is the working
mechanism, ablations welcome); OLMo tested at one scale (1B, stage 1). All
scripts, pre-registrations and negative results in the repo. **If you have
checkpoint suites (Amber, SmolLM, OLMo-7B…): please run the probe and
report — that's the fastest way to map where this lives.**

### 5 · Provenance
Falsification-first research line (pre-registered endpoints, published
kills). The same instrumentation that failed to find "reasoning" signatures
in representation dynamics — published alongside — found this instead.

---
