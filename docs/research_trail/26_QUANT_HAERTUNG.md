# Quant×Lock-in — Härtung: GGUF bestätigt, 410m-Vorhersage nur teilweise

**Stand: 2026-07-15 · `gguf_quant_results.json` + `quant_lockin_410m.json`.**

## Härtung 1 · Echte GGUF-K-Quants (Q4_K_M via llama.cpp): **BESTÄTIGT**

| Checkpoint 160m | PPL f16 | PPL Q4_K_M | ΔlnPPL |
|---|---|---|---|
| step 84000 | 1.655 | 1.716 | **0.036** |
| step 143000 | 1.679 | 1.937 | **0.142** |

**Ratio 143k/84k = 3.94** (vorregistriertes Kriterium ≥2). Der RTN-Befund
gilt auch mit Produktions-Quantisierung. Bonus: f16-Qualität bei 84k sogar
minimal besser als final — die Deployment-Regel („nicht der letzte
Checkpoint fürs Q4") steht damit auf echten GGUF-Zahlen.
(Nebenkosten der Härtung: transformers-5-Umbenennung `rotary_pct→
partial_rotary_factor` brach den llama.cpp-Konverter — lokal gepatcht,
`conversion/gptneox.py`, dokumentiert.)

## Härtung 2 · 410m-Replikation: **Mechanismus repliziert, Kriterium verfehlt — ehrlich: teilweise**

| 410m | 64k | 76k | 84k | 92k | 108k | 128k | 143k |
|---|---|---|---|---|---|---|---|
| eff. Rang | 12.7 | 12.9 | 12.8 | 12.6 | 11.1 | 8.6 | 8.1 |
| Int4-Schaden | 0.16 | 0.40 | 0.50 | 0.61 | 0.66 | 1.30 | 1.11 |

- **Mechanismus repliziert:** Spearman(Rang, Schaden) = **−0.857** (160m:
  −0.964); Schaden steigt, wo der Rang fällt (Kompression ab ~108k beim 410m).
- **Niveau-Vorhersage hält:** finaler 410m-Schaden 1.11 vs. 160m 3.71 —
  das größere Modell mit höherem Restrang (8.1 vs. 1.6) ist ~3× robuster,
  konsistent mit „Schaden wächst mit Kompressionstiefe".
- **ABER das vorregistrierte Zahlenkriterium ist verfehlt:** relative
  pre/post-Änderung 159 % — verlangt waren <110 % (halb von 160m). Das
  Kriterium war schlecht gewählt: die Ratio-von-Relativen ist nicht
  skalenfrei (der 410m-Pre-Schaden ist selbst winzig, was die Relation
  aufbläht). Wir werten NICHT um: **formal teilweise bestätigt**, mit
  dokumentierter Kriterium-Lektion für die nächste Präregistrierung
  (Niveau- statt Verhältnis-Kriterien).

## Konsolidierter Stand des Quant-Befunds

Über 2 Modellgrößen, 2 Quantisierungsverfahren (RTN, Q4_K_M), 7+7+2
Checkpoints: **Die Post-Lock-Kompression des Fahrplans geht einher mit
wachsender 4-Bit-Fragilität (ρ −0.86…−0.96), unsichtbar für Loss und
Benchmarks.** Die Rang-Kurve ist damit als praktischer Checkpoint-Wähler
für Q4-Deployment einsatzbereit — der erste belegte Produktnutzen der
Telemetrie. Offene Härtungen: englischer Sondenkorpus · Kausalrichtung
(ist die Kompression Ursache oder Ko-Symptom?) · dritter Maßstab (70m/1b).
