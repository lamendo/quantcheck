# Einrast-Studie (A-Paket) — Skalen- und Daten-Härtung der Entstehungsbefunde

**Stand: 2026-07-14 · Läufe: Pythia-160m (Standard, aus 14), Pythia-410m,
Pythia-160m-deduped (anderes Trainingskorpus!) · je 12 Checkpoints step0–143000
· `pythia_{emergence,410m,deduped}.json` · Seeds-Variante blockiert
(Seed-Repos nur .bin; torch<2.6-CVE-Sperre — dokumentiert, nicht umgangen;
Daten-Variante ist der härtere Test).**

## Verdikte

### H-A2 · Attraktor: **STARK GESTÜTZT — jetzt über DATEN-Variation**
Das deduped-Modell (gleiche Architektur, entdupliziertes = anderes
Trainingskorpus) endet auf **exakt derselben Fahrplanform wie das
Standard-160m: r = 1.00** (Trio: 0.936) — und auf demselben End-Rang
(1.63 vs. 1.62!). Sechster konvergierender Lauf, erstmals über die
Daten-Achse. Die Form ist damit gegen Architekturfamilie, Größe,
Präzision, Runtime UND Trainingsdaten-Zusammensetzung robust — die
Attraktor-These hat ihre bisher härteste Prüfung bestanden.

### H-A4 · Rogue↔Warmup: **GESTÜTZT (3/3 Läufe)**
Ausbruch beginnt in allen Läufen nach Warmup-Ende (1430) und peakt bei
step 4000: 160m **19** · 410m **22** · deduped **20** Dims; danach Pruning
(→6/12–13/9). Timing und Gestalt skalen- und datenstabil.
Neues Detail: Der Ausbruch VERFORMT die Fahrplanform transient
(deduped: r_trio fällt bei 4k auf 0.16!, Standard auf 0.80) — die Form
taucht durch und kehrt zum Attraktor zurück.

### H-A3 · Reihenfolge: **GESTÜTZT in präzisierter Form**
Die Kompression hat ZWEI Phasen, in allen drei Läufen: (1) Rogue-Burst
BLÄHT den Rang erst auf (160m 9.2 · 410m 20.0 · deduped 9.0), dann
Relaxation; (2) die SPÄTE, eigentliche Kompression startet einheitlich im
64k–128k-Fenster (160m-Feinsweep: exakt am Form-Lock 84k). **In keinem
Lauf beginnt die späte Kompression vor dem Lock-Fenster.** Die
Kausal-Lesart „Einrasten schaltet Kompression frei" überlebt.

### H-A1 · Timing-Gesetz: **INNERHALB DER PYTHIA-SUITE NICHT ENTSCHEIDBAR**
Alle Pythia-Modelle teilen denselben Schedule (143k Steps, 300B Tokens) —
relativer und absoluter Fortschritt fallen zusammen. Beobachtbar ist nur:
Das Reorganisations-Fenster liegt über Größe und Daten an derselben Stelle.
Entscheidung braucht eine Suite mit anderer Trainingslänge → **OLMo-
Checkpoints als Kandidat** (vorgemerkt, nicht gestartet).

## Neuer Skalen-Befund

**End-Rang ist größenabhängig, nicht datenabhängig:** 160m endet bei
1.62/1.63 (Standard/deduped identisch!), 410m bei **7.67**. Das größere
Modell behält einen reichhaltigeren Fahrplan — die Kompressions-*Tiefe*
skaliert mit der Modellgröße, das Kompressions-*Timing* nicht. Anschluss-
Hypothese (offen): Endrang ~ f(Modellbreite) — prüfbar mit 70m/1b/1.4b
aus derselben Suite, gleiche Pipeline.

## Stand des Entstehungs-Bilds nach dieser Studie

> LM-Training zieht Transformer unabhängig von Architektur, Größe und
> Datenmischung in dieselbe Tiefen-Organisationsform. Der Weg dorthin hat
> eine feste Ereignisfolge: früher Rogue-Ausbruch nach Warmup (mit
> transienter Form-Deformation), Relaxation, langes Plateau, spätes
> Einrast-Ereignis (Form-Lock), das eine kontinuierliche, größenabhängig
> tiefe Rang-Kompression freischaltet. Endogen (kein Schedule-Gegenstück),
> skalenstabil, datenstabil. Offen: absolute vs. relative Taktung (OLMo)
> und der Mechanismus des Einrastens selbst.
