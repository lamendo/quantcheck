# 31 — Härtungstest Übertraining: TinyLlama-1.1B → KILL

**Datum:** 2026-07-29 · **Skript:** `topo_steering/tinyllama_haertung.py` ·
**Ergebnisse:** `artifacts/topo_steering/tinyllama_results.json`

## Frage & Vorregistrierung

Pythia-1b (300B Tokens) zeigte keine späte Kompression/Fragilität → These:
Übertraining erzeugt beides auch bei moderner Rezeptur. TinyLlama-1.1B
(3T Tokens, ~2.700 Tok/Param, 7 öffentliche Checkpoints 105B–3T).
Kriterien (vor dem Lauf): K1 ρ(Tokens, Rang) ≤ −0.7 · K2 ρ(Tokens, Int4-
Schaden) ≥ +0.7 · K3 ρ(Rang, Schaden) ≤ −0.7 · BESTÄTIGT nur bei K1∧K2∧K3.

## Ergebnis: KILL (alle drei Kriterien klar verfehlt)

| Tokens (Mrd.) | NLL fp32 | eff. Rang | Δ Int4 |
|---|---|---|---|
| 105 | 6.12 | 4.51 | +0.018 |
| 503 | 8.93 (*Ausreißer*) | 8.78 | +0.515 |
| 1000 | 5.34 | 4.53 | −0.219 |
| 1500 | 4.91 | 4.50 | −0.139 |
| 2000 | 5.07 | 4.64 | −0.065 |
| 2500 | 4.37 | 5.60 | −0.027 |
| 3000 | 4.22 | **2.49** | +0.156 |

K1 = −0.25 · K2 = −0.04 · K3 = +0.18 → **„Befund bleibt (vorerst)
Pythia-gebunden."** Robustheit ohne den 503B-Ausreißer (bekannte Config-
Macken der TinyLlama-Zwischencheckpoints): K1 −0.03 · K2 +0.43 · K3 −0.31 —
ebenfalls KILL.

## Ehrliche Nuancen

1. **TinyLlama bleibt trotz extremen Übertrainings Int4-robust.** Der Schaden
   liegt von 1T–3T praktisch bei null (−0.22 … +0.16); kein Pythia-artiger
   Fragilitäts-Anstieg. Die Übertraining-These ist damit in ihrer starken
   Form tot.
2. **Eine schwache Spur existiert:** In der stabilen Phase (≥1T) steigt der
   Schaden perfekt monoton (ρ=1.0 über 5 Punkte, aber Magnitude winzig), und
   der 3T-Endpunkt hat den niedrigsten Rang (2.49) UND den höchsten späten
   Schaden. Aber: K3 in dieser Phase = −0.1 — der Rang trägt den Anstieg
   NICHT. Einzelpunkt, keine Kette.
3. **Konsequenz für den Mechanismus-Anspruch:** Rang↔Schaden ist eine
   *Pythia-interne* Korrelation (−0.96/−0.86), kein familienübergreifendes
   Gesetz. Moderne Rezeptur (TinyLlama: anderes LR-Schema, Daten, FlashAttn)
   erreicht offenbar auch bei 2.700 Tok/Param nicht das fragile Regime.

## Konsequenz für die Publikation

- Rahmung explizit **Pythia-Familie**; TinyLlama als ehrliche negative
  Replikation MIT ins Paket (stärkt die Glaubwürdigkeit und ist selbst
  informativ: „nicht jedes Übertraining macht fragil — miss nach").
- Der Nutzwert der Probe bleibt unberührt: Sie sagt pro Suite/Checkpoint,
  ob man im fragilen Regime ist — TinyLlama-Nutzer bekommen „alles ok",
  Pythia-Nutzer „nimm 84k statt final". Genau dafür ist sie da.
