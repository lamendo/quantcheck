# Announcement drafts (for Jan's review — do not publish before sign-off)

## 1 · Hugging Face community post (day 0, with repo link)

**Title: Don't quantize your last checkpoint — a label-free probe that tells
you which one to pick**

Late in pretraining, models can keep identical benchmark scores while
becoming dramatically more fragile to 4-bit quantization. On Pythia-160m the
final checkpoint takes **3.9× more GGUF-Q4_K_M damage** than a checkpoint
from the same run that scores the same on benchmarks.

We traced this to a measurable internal event: after an endogenous "lock-in"
transition, the mean depth-update of the residual stream collapses onto ever
fewer directions. That **rank compression correlates with Int4 damage at
ρ ≈ −0.9 in three model families** (Pythia across 4 scales, OLMo-2-1B, and
— as a pre-registered *negative* — TinyLlama, which never enters the fragile
regime despite 2,700 tokens/param of overtraining).

So it's not a universal law — it's a **measurable regime**, and the probe
(forward passes only, no labels, no benchmarks) tells you which one your
model is in *before* you press it.

Also in the repo: our pre-registered *failed* attempt to repair fragility
post-hoc (rank-restoring finetune — rank is steerable, quality dies), which
is exactly why checkpoint choice matters. All pre-registrations, all
negatives, an interactive visual explainer, and scripts that run on a single
8 GB consumer GPU. If you have checkpoint suites (Amber, SmolLM, OLMoE-7B…):
please run the probe and report — mapping where this lives is the point.

## 2 · r/LocalLLaMA (day 1)

**Title: PSA for quantizers: the last checkpoint can be the worst one to
quantize — same benchmarks, 4× the Q4 damage. Cheap probe inside.**

Body: short version of the above, lead with the Pythia table, link repo +
HF post. One honest paragraph on limits (RTN+GGUF tested, AWQ/GPTQ not;
three families mapped; correlation ≠ proven causation — but the failed
repair experiment is causal evidence that compression is load-bearing).

## 3 · X thread (day 2, 6 tweets)

1. Your benchmarks say the model is done. Its interior disagrees. 🧵
2. [Terrain image from explainer] Training a small LM: benchmarks flat from
   step 13k. Inside, at step 84k, the model "locks in" and starts compressing
   its depth-schedule onto fewer directions.
3. That compression predicts 4-bit damage: ρ ≈ −0.9, three families.
   Final checkpoint = up to 4× worse GGUF-Q4 than its benchmark-twin sibling.
4. Not a universal law: TinyLlama (3T tokens!) never enters the regime.
   That's why you measure instead of assume. Probe = forward passes only.
5. We tried to repair it post-hoc (pre-registered): you can re-inflate the
   rank, but quality dies. The compression IS the capability. Pick your
   checkpoint; don't patch it.
6. Repo + interactive explainer + every pre-registration incl. the kills: [link]

## Channel order & timing

GitHub public → HF post (same day) → r/LocalLLaMA (+1 day) → X (+2 days).
Replication issues pre-filed: Amber, SmolLM, OLMoE-7B, Pythia-2.8b (fp16).

## Open items before any of this goes out

- [x] Repo name: **quantcheck** (entschieden 2026-08-01)
- [x] License: Apache-2.0 (entschieden 2026-08-01)
- [ ] Attribution line (proposal: "Jan R. — lamendo") (Jan)
- [ ] OLMoE verdict integrated (study 36, running)
- [ ] Jan's read-through of README + PREREG + this file
