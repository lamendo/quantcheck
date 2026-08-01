# V2-Kerntest R1 — Lock-in ≠ Benchmark-Reife (Negativ, mit Wendung)

**Stand: 2026-07-15 · Code `v2_kerntest.py` · Daten: öffentliche per-Checkpoint-
Zero-Shot-Evals (EleutherAI/pythia GitHub) × unsere Emergenz-Kurven ·
Kosten: 0 GPU-Minuten, reine Analyse.**

## Ergebnis

| Lauf | Median-Reife-Punkt der Benchmarks (90 % der Spanne) | Lock-in |
|---|---|---|
| 160m | **step 13 000** | step ~84 000 |
| 160m-deduped | **step 13 000** | 64k–128k-Fenster |
| 410m | step 63 000 (Streuung 13k–143k) | 64k–128k-Fenster |

T2 (Lambada-Segmentsteigungen 160m): 53–73k **+0.14**, 73–93k **−0.03**,
danach negativ — **kein Knick, keine Beschleunigung am Lock-in**; die
Benchmark-Kurven sind dort längst flach.

> **Verdikt: Die naive V2-These („Lock-in-Timing sagt Benchmark-Reife
> voraus") ist in Runde 1 FALSIFIZIERT.** Die Standard-Benchmark-Reife der
> kleinen Pythias ist bei ~13k Steps erreicht — das Einrasten bei 84k und
> die anschließende Kompression passieren in einer Phase, in der Benchmarks
> nichts mehr sehen.

## Die Wendung (neue, schärfere Frage — nicht heute behauptet)

Das Lock-in ist damit ein reales, endogenes, mehrfach repliziertes Ereignis
**ohne Benchmark-Korrelat** — es misst etwas, das Zero-Shot-Accuracy nicht
misst. Kandidaten, WOFÜR die späte Reorganisation zählt (jeweils prüfbar):
Kalibrierung · Quantisierungs-Robustheit später Checkpoints (Anschluss an
den Rogue-Befund!) · Finetuning-Startpunkt-Qualität. Erst wenn eine dieser
Größen am Lock-in reagiert, hat die Telemetrie ihren Nutzen-Beweis.

## Ehrliche Grenzen

90 %-Spannen-Kriterium ist grob; kleine Modelle sättigen leichte Zero-Shot-
Benchmarks früh (Decken-Effekt); schwere Benchmarks sind auf diesen Skalen
Rauschen. Der Test war fair, aber benchmark-limitiert — die Wendungs-Fragen
oben umgehen genau diese Grenze.

## Was von V2 übrig bleibt

- **Unverändert nützlich (brauchten den Benchmark-Link nie):**
  Rogue-Inventar als Quantisierungs-Risiko-Check · Regressions-/Integritäts-
  QA (Form-Vergleich vor/nach Eingriff) · Ereignis-Detektion in eigenen Läufen.
- **Zurückgestuft:** „Reifegrad-Vorhersage" als Produktversprechen — bis
  eine der Wendungs-Größen den prädiktiven Nutzen zeigt.
