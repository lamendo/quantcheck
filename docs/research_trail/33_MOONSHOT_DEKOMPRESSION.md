# 33 — Moonshot „Decompress, then Quantize": Präregistrierung

**Datum:** 2026-07-29 (Präregistrierung VOR allen Läufen) ·
**Skripte:** `topo_steering/decompress_kg0.py` (KG-D0), KG-D1/D2 folgen nur bei PASS.

## These

Die Rang-Kompression des Tiefen-Fahrplans ist nicht nur *Korrelat*, sondern
*Ursache* der Int4-Fragilität (K3 = −0.96/−0.86/−0.87 über Pythia-160m/410m/
OLMo-2). Dann muss gelten: Ein kurzer Post-Training-Schritt, der den
Fahrplan-Rang eines FERTIGEN Checkpoints wieder anhebt („Dekompression"),
senkt dessen Quantisierungs-Schaden — ohne die fp32-Qualität zu opfern.

Falls wahr: universell anwendbarer Quantisierungs-Vorbereitungs-Schritt für
Modelle OHNE Checkpoint-Historie (der eigentliche Produktfall — niemand hat
Llama-Zwischencheckpoints); „QAT ohne QAT".
Falls falsch: Kompression ist Symptom, nicht Ursache — beantwortet unsere
offene Kausalfrage. Beides ist ein Ergebnis; nichts davon wird versteckt.

## Aufbau

- **Subjekt:** Pythia-160m, finaler Checkpoint step143000 (fragilster
  bekannter Punkt: Rang 1.62, Int4-Schaden 3.71 auf DE-Probe).
- **Intervention:** kurzes Finetune (~1–2 M Tokens, englischer Fließtext,
  disjunkt von allen Proben), Loss = LM-Loss − λ·effRang(μ_batch), wobei
  μ_batch = mittlerer Tiefen-Update-Vektor des Batches und
  effRang = (Σσ)²/Σσ² (differenzierbar via svdvals). Embeddings eingefroren
  (der Fahrplan lebt in den Blöcken; halbiert Optimizer-Speicher).
- **Messgrößen** (identische Pipeline wie Serie 25/26: DE-Probe, fp32,
  RTN-Int4): nll_fp32, eff_rang, delta_int4. Baseline wird im selben Lauf
  neu gemessen (keine Zahlen-Übernahme).

## KG-D0 — Machbarkeit (dieser Lauf)

λ-Fächer {0.03, 0.1, 0.3} × 300 Schritte (deklarierte Feasibility-Suche,
KEIN Ergebnis-Shopping: KG-D1 läuft mit dem einen gewählten λ neu).

**PASS genau dann, wenn mindestens ein λ erreicht:**
1. eff_rang ≥ 3.0 (Baseline ~1.6 — Rang ist steuerbar), UND
2. nll_fp32 ≤ Baseline × 1.01 (Qualität unangetastet).

**KILL:** kein λ schafft beides → Rang ist nicht billig anhebbar oder nur
gegen Qualität → Moonshot in dieser Form beerdigt.

Sekundär (rein deskriptiv, KEIN D0-Kriterium, um Bias zu vermeiden):
delta_int4 wird mitgemessen und berichtet, entscheidet aber erst KG-D1.

## KG-D1 — Die eigentliche Wette (nur bei D0-PASS)

Gewähltes λ, frischer Lauf + **Kontrollarm: identisches Finetune mit λ=0**
(gleiche Daten/Schritte/Seed — isoliert den Regularisierer vom bloßen
Weitertrainieren).

**BESTÄTIGT genau dann, wenn:**
1. delta_int4(dekomprimiert) ≤ 0.5 × delta_int4(Baseline step143000), UND
2. delta_int4(dekomprimiert) < delta_int4(Kontrollarm λ=0) − 0.5, UND
3. nll_fp32 ≤ Baseline × 1.01.

**KILL:** Kriterium 1 oder 2 verfehlt → Kompression ist Symptom, nicht
Ursache; als Kausal-Ergebnis dokumentieren (eigenständig publikabel).

## KG-D2 — Transfer (nur bei D1-BESTÄTIGT)

OLMo-2-1B final (2.8×-Regime, moderne Rezeptur) + GGUF-Q4_K_M-Replikation
statt RTN-Proxy. Kriterien analog D1 (≤ 0.5× + Kontrollarm + Qualität).

## KG-D0 Ergebnis (2026-07-29): formal KILL — Confound identifiziert

| Arm | nll_fp32 | eff_rang | delta_int4 (deskriptiv) |
|---|---|---|---|
| Baseline 143k | 4.813 | 1.68 | 3.71 |
| λ=0.03 | 5.407 (+12 %) | 1.58 | 2.88 |
| λ=0.1 | 5.339 (+11 %) | 1.76 | 3.00 |
| λ=0.3 | 5.483 (+14 %) | **3.02** | 3.11 |

Diagnose: (1) **Rang ist steuerbar** (λ=0.3 erreicht das Rangziel in 300
Schritten). (2) NLL-Kriterium scheiterte am **Anker, nicht am Regularisierer**
— auch λ=0.03 (Rang unbewegt) verliert 12 %: englischer Wikitext-Anker gegen
DE-Probe = reiner Distributions-Drift (Designfehler dieses Laufs).
(3) Nebenbefund: delta_int4 sinkt in ALLEN Armen (3.71→2.9–3.1), auch ohne
Rang-Anhebung — per Präreg erst in KG-D1 bewertbar.

**Eskalation D0b (deklarierter Confound-Repair, Kriterium unverändert):**
Anker = Pile-Stichprobe (eigene Trainingsverteilung, NeelNanda/pile-10k),
lr 1e-5, 1000 Schritte, λ ∈ {0, 0.3, 1.0} (λ=0 = Drift-Boden-Kontrolle).
Danach gilt das D0-Verdikt endgültig. Skript `decompress_kg0b.py`.

## KG-D0b Ergebnis (2026-07-29): ENDGÜLTIG KILL

Pile-Anker, lr 1e-5, 1000 Schritte (4.1 M Tokens):

| Arm | nll_fp32 | eff_rang | delta_int4 (deskriptiv) |
|---|---|---|---|
| Baseline 143k | 4.813 | 1.68 | 3.71 |
| λ=0 (Kontrolle) | 4.800 (−0.3 %) | 1.58 | 3.26 |
| λ=0.3 | 5.058 (**+5.1 %**) | 3.30 ✓ | 3.42 |
| λ=1.0 | 7.297 (**+51.6 %**) | 9.66 ✓✓ | 2.07 |

Der Anker war sauber (λ=0: NLL sogar minimal besser) — der Qualitätsverlust
ist damit eindeutig dem Rang-Term zuzurechnen. Es existiert eine glatte
Trade-off-Kurve: Rang ist beliebig anhebbar (bis 9.66!), aber JEDER Punkt,
der das Rangziel erreicht, verletzt das +1-%-Qualitätsband. Per
Präregistrierung gilt: **KILL, Moonshot in dieser Form beerdigt.**

## Interpretation (der eigentliche Erkenntnisgewinn)

1. **Die Post-Lock-in-Kompression ist nicht kosmetisch — sie ist tragend.**
   Der niedrige Fahrplan-Rang lässt sich nachträglich nicht aufblasen, ohne
   die Funktion zu beschädigen: Die späte Kompression IST offenbar die Form,
   in der das Modell seine Fähigkeiten spät im Training kodiert. „Symptom
   vs Ursache" bekommt eine dritte Antwort: **verwachsen.**
2. **Kausal-Hinweis bleibt:** λ=1.0 halbiert den Int4-Schaden (3.71→2.07)
   bei Rang 9.66 — die Richtung Rang→Robustheit stimmt. Aber der Preis
   (+52 % NLL) zeigt: Man schützt das Modell, indem man es zerstört.
   Kein nutzbarer Pfad.
3. **Konsequenz fürs Produkt/Paper:** Der einzige bekannte Weg zu
   „Fähigkeit UND Quant-Robustheit" bleibt die **Checkpoint-Wahl** (bzw.
   Trainings-Rezeptur à la TinyLlama) — genau die Regel, die quantcheck
   liefert. Der Kill STÄRKT die Publikation: „nachträglich reparieren geht
   nicht (wir haben es versucht, Präreg + Zahlen im Repo) — also miss und
   wähle richtig."
4. **Nicht weiterverfolgt (bewusst):** Training-Zeit-Regularisierer von
   Beginn an (echte QAT-Alternative) — plausibel, aber jenseits der
   8-GB-Skala für relevante Modelle; nur mit Partner/Compute sinnvoll.

## Deklarierte Grenzen

Eine Modellgröße in D0/D1 (160m); RTN-Proxy bis D2; DE-Probe als
Primärmaß (Serie-konsistent); 300 Schritte sind eine Untergrenze — ein
D0-KILL bei allen λ wird einmal mit 1000 Schritten und λ=1.0 gegengeprüft
(deklariert als einzige erlaubte Eskalation), danach gilt das Verdikt.
