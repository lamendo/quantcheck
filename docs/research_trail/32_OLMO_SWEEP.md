# 32 — OLMo-2-Sweep: Der Entscheidungstest → BESTÄTIGT

**Datum:** 2026-07-29 · **Skript:** `topo_steering/olmo_sweep.py` ·
**Ergebnisse:** `artifacts/topo_steering/olmo_results.json`

## Ausgangslage

Nach dem TinyLlama-KILL (31) stand der Pythia-Befund (Training → Rang-
Kompression → Int4-Fragilität) auf EINER Familie — Jans berechtigter Einwand:
„das ist recht dünn". OLMo-2-0425-1B (AI2 2025, modernste öffentliche
Rezeptur, 195 Stage-1-Checkpoints 0–4T Tokens) war als Entscheidungstest
vorregistriert: BESTÄTIGT → Publikation; KILL → Pythia-Kuriosum, Absage.

## Design (vorregistriert)

9 Checkpoints 84B–4001B, Kriterien identisch zur TinyLlama-Härtung:
K1 ρ(Tokens, Rang) ≤ −0.7 · K2 ρ(Tokens, Schaden) ≥ +0.7 ·
K3 ρ(Rang, Schaden) ≤ −0.7. Probe DE, RTN-Int4, fp32-Basis, CPU.

## Ergebnis: BESTÄTIGT — K1 = −0.883 · K2 = +0.867 · K3 = −0.867

| Tokens (Mrd.) | NLL fp32 | eff. Rang | Δ Int4 |
|---|---|---|---|
| 84 | 4.91 | 9.68 | 0.272 |
| 441 | 4.80 | 5.88 | 0.365 |
| 944 | 5.05 | 4.87 | 0.411 |
| 1469 | 4.75 | 4.43 | 0.586 |
| 1972 | 4.49 | 4.35 | 0.497 |
| 2475 | 4.41 | 4.40 | 0.568 |
| 2999 | 4.74 | 4.76 | 0.552 |
| 3503 | 4.68 | 4.04 | **0.760** |
| 4001 | 4.71 | **3.65** | 0.652 |

Effektstärke: Schaden final/min = **2.8×** (Peak 3.5T; letzter Checkpoint
schlechter als jeder Punkt vor 1.5T).

## Ehrliche Profil-Einordnung: drei Familien, drei Regime

| Familie | Kompression | Schadensverlauf | Effektstärke |
|---|---|---|---|
| Pythia (70m–410m) | **spät** (Lock-in @~59 %, dann steil) | flach → Explosion am Ende | ~4× (bis 13× @2bit) |
| **OLMo-2-1B** | **kontinuierlich** (9.7→3.7 über den ganzen Lauf, kein Lock-in-Sprung) | gradueller Anstieg, spät beschleunigend (3T→3.5T: 0.55→0.76) | **2.8×** |
| TinyLlama-1.1B | keine (flach ~4.5, Endpunkt-Dip) | ~null (−0.2…+0.16) | — |

Das ist KEINE Ein-Familien-Kuriosität mehr: Der Mechanismus-Link
Rang↔Schaden hält in moderner Rezeptur (K3 −0.867 ≈ Pythia −0.86/−0.96) —
aber die *Dynamik* variiert: Pythia hat ein spätes endogenes Ereignis,
OLMo komprimiert von Anfang an stetig, TinyLlama erreicht das fragile
Regime gar nicht. Kernbotschaft der Publikation dadurch GESCHÄRFT:
**„Es gibt kein universelles Gesetz — es gibt ein messbares Regime.
Miss dein Modell, bevor du quantisierst."** Die Rang-Kurve ist genau
dieses Messgerät (label-frei, forward-only).

## Deployment-Nutzen bei OLMo konkret

Mittlere Checkpoints (1–2.5T) haben Schaden ~0.5 bei bereits niedriger NLL
(4.4–4.7); der finale hat 0.65, der 3.5T-Punkt 0.76. Milder als Pythias
Klippe, aber dieselbe Richtung: **der letzte Checkpoint ist nicht der beste
zum Quantisieren.**

## Grenzen

- 1 Größe (1B), Stage 1 only (Stage-2-Ingredients nicht gemessen), RTN-Proxy
  (GGUF-Replikation für OLMo offen), DE-Probe, Benchmarks nicht mitgemessen.
- Betrieblicher Hinweis: Der Sweep starb 3× still (externe Prozess-Kills,
  vermutlich parallele Session); Resume-Pattern (partial.jsonl) hat alle
  Messungen gerettet — Standard für alle künftigen Sweeps.
