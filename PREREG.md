# Pre-registrations & verdicts

Every claim in this repo was pre-registered: endpoints and kill criteria were
written down **before** the runs. This file is the ledger — including the
negatives. Full research trail (German): lamendo research — "Representation Observatory"
series, studies 07–36.

| # | Study | Pre-registered criterion | Verdict |
|---|---|---|---|
| 1 | Lock-in ↔ Int4 damage (Pythia-160m) | Damage rises ≥20 % post-lock-in; Int8 control flat | **CONFIRMED** — +220 %, Spearman(rank, damage) = −0.964 |
| 2 | GGUF replication (Q4_K_M) | Final/best checkpoint damage ratio ≥ 2× | **CONFIRMED** — 3.94× |
| 3 | 410m scale replication | Pre-declared: flatter rise than 160m | Mechanism replicated (ρ = −0.857); the *numeric* criterion (ratio-of-relatives) misfired — documented as a criterion-design lesson, verdict "partial" |
| 4 | 70m / 1b scale chain | Rank end-level predicts damage ordering | **CONFIRMED** — 70m (rank 1.13 → damage 8.15) … 1b (7.02 → 0.55) |
| 5 | Bit sweep (RTN 2–8 bit) | Effect exists at every usable depth | **CONFIRMED** — amplifies toward low bits (13× at 2-bit) |
| 6 | EN probe | Same curve shape as DE | **CONFIRMED** (+283 % pre/post) |
| 7 | TinyLlama overtraining (7 ckpts, 3T tokens) | K1/K2/K3 Spearman ≥ \|0.7\| | **KILLED** — all three missed; damage ≈ 0 from 1T–3T. Not a universal law. |
| 8 | OLMo-2-1B modern recipe (9 ckpts, 4T tokens) | Same K1/K2/K3 | **CONFIRMED** — −0.883 / +0.867 / −0.867; continuous-drift profile (2.8×), not Pythia's late cliff |
| 9 | Post-hoc "decompression" fix (rank-restoring finetune) | Rank ≥ 3.0 AND NLL ≤ +1 % | **KILLED** (2 designs incl. clean in-distribution anchor): rank is steerable (up to 9.66) but every rank-effective setting breaks the quality band → late compression is load-bearing. **You cannot repair it after the fact; you must pick the right checkpoint.** |
| 10 | OLMoE-1B-7B MoE suite (8 ckpts, 5T tokens) | K-M1 router-lock / K-M2 compression / K-M3 rank↔damage / K-M4 depth-schedule attractor r ≥ 0.9 | **SUSPENDED at 4/8** (local machine constraints, honest partial data in `results/olmoe_partial_4of8.json`). Early read: tiny Int4 damage so far; mean routing entropy pinned at ln(64) — the pre-declared caveat held: mean usage measures the load-balancer, not specialization. Completing this suite is an explicit replication ask. |

Notes on honesty conventions used throughout: level criteria instead of
ratio-of-relatives after study 3's lesson; declared escalations only (one per
study, written before the rerun); measurement-design flaws are reported as
such, not silently repaired (see studies 3, 9, 10 notes in the German docs).
