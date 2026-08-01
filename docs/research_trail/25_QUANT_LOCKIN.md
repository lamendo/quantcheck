# Quantisierung × Lock-in — der erste Nutzen-Beweis der Telemetrie

**Stand: 2026-07-15 · Code `quant_lockin.py` · Ergebnisse
`quant_lockin_results.json` · Pythia-160m, 7 Checkpoints 64k–143k, alle aus
Cache (0 Downloads) · Endpunkt vorregistriert: Int4-RTN-Schaden (ΔNLL,
Sondenkorpus) pre vs. post Lock-in, Kriterium ≥20 % relative Änderung ·
Int8-Kontrolle.**

## Ergebnis: **LOCK-IN-RELEVANT — Kriterium um das 11-Fache übertroffen**

| Step | eff. Rang μ | Int4-Schaden (ΔNLL) | Int8 (Kontrolle) |
|---|---|---|---|
| 64000 (pre) | 5.20 | 0.55 | ~0 ✓ |
| 76000 (pre) | 5.33 | 0.60 | ~0 ✓ |
| 84000 (**Lock**) | 4.23 | 0.78 | ~0 ✓ |
| 92000 | 3.20 | 0.92 | ~0 ✓ |
| 108000 | 2.20 | 1.50 | ~0 ✓ |
| 128000 | 1.80 | 2.11 | ~0 ✓ |
| 143000 (final) | 1.62 | **3.71** | ~0 ✓ |

Pre-Mittel 0.64 vs. Post-Mittel 2.06 → **+220 %** (Kriterium: ≥20 %).
Der Int4-Schaden beginnt exakt am Lock-in zu steigen und versechsfacht sich
bis Trainingsende — während die Benchmarks in derselben Phase flach sind
(24_V2_KERNTEST).

## Der mechanistische Link

**Spearman(eff. Rang, Int4-Schaden) = −0.964** über die 7 Checkpoints:
Je weiter die Post-Lock-Kompression fortschreitet, desto fragiler wird das
Modell unter 4-Bit-Quantisierung. Konsistente Lesart: Die Konzentration der
Verarbeitung auf wenige dominante Richtungen (Rang 5.3 → 1.6) macht die
Repräsentation empfindlich gegen Gewichts-Rundung — die späte Trainingsphase
kauft (benchmark-unsichtbare) Struktur zum Preis von Quantisierungs-Robustheit.

## Praktische Konsequenz (Hinweis-Niveau, sofort prüfbar in der Praxis)

Für Q4-Deployment kleiner Modelle ist der LETZTE Checkpoint möglicherweise
der SCHLECHTESTE: Bei step 84k–92k ist der Int4-Schaden 4-mal kleiner als
bei 143k, bei praktisch gleicher Benchmark-Leistung (Reife war ~13k).
**Unsere Telemetrie (Rang-Kurve) zeigt an, wann dieser Kipppunkt beginnt —
eine Information, die weder Loss noch Benchmarks liefern.** Das ist der
erste demonstrierte prädiktive Nutzen des „EKGs".

## Grenzen (deklariert)

1. RTN-Int4 ist ein Proxy — moderne K-Quants/AWQ mildern; der TREND muss
   dort repliziert werden (nächster Schritt: GGUF-Q4 via llama.cpp auf 2–3
   Checkpoints).
2. Sondenkorpus deutsch, Pile englisch — ΔNLL ist intern kontrolliert
   (fp32 vs int4 je Checkpoint), aber englische Replikation steht aus.
3. Ein Modell, eine Größe — 410m-Replikation billig möglich (Vorhersage:
   flacherer Anstieg, da Rang nur bis 7.7 fällt!). Das ist eine ECHTE
   Vorab-Vorhersage aus dem Mechanismus-Link.
