# Die Entstehungsgeschichte — Pythia-Checkpoint-Reihe: der Fahrplan ist ein ATTRAKTOR

**Stand: 2026-07-13 · Code `pythia_checkpoints.py` · Ergebnisse
`pythia_emergence.json` · Pythia-160m, 12 Checkpoints step0→143000,
identische 31 Prompt-Texte, prefill-only.**

## Die Zeitreihe

| Step | Form r(trio) | Krümmung r(final) | Rogue-Dims | eff. Rang μ |
|---|---|---|---|---|
| 0 (Init!) | **0.95** | +0.64 | 0 | 8.0 |
| 1000 | 0.92 | +0.44 | 0 | 7.9 |
| 2000 | 0.83 | +0.46 | **6** | 9.2 |
| 4000 | 0.80 | +0.35 | **19** | 8.0 |
| 8000 | 0.70 | **−0.07** | 12 | 7.0 |
| 16000–64000 | 0.82–0.86 | **−0.38…−0.11** | 6–10 | 5.1–5.4 |
| 128000 | **0.94** | **+0.99** | 6 | **1.8** |
| 143000 (final) | **0.94** | +1.00 | 6 | **1.6** |

## Befund 1 · Die Form-Geschichte korrigiert den gestrigen Verdikt — präzisiert ihn aber wunderbar

**Pythia trägt die Trio-Form schon bei Initialisierung (r=0.95)** — im Kontrast
zum Random-Qwen von gestern (r=0.24). Das „TRAINING"-Verdikt aus Dokument 13
war zu einfach. Das kombinierte Bild beider Experimente:

> **Die universelle Fahrplanform ist ein ATTRAKTOR des Trainings, kein reiner
> Architektur- und kein reiner Lern-Effekt.** Manche Initialisierungen starten
> nahe der Form (GPT-NeoX/Pythia: 0.95), andere fern (Qwen2-Init: 0.24) —
> aber ALLE trainierten Endpunkte konvergieren auf sie (jetzt VIER Familien:
> Qwen2.5, SmolLM2, Qwythos, Pythia — final r=0.94). Und die Reise dorthin ist
> nicht monoton: Das Training VERBIEGT die Form in der mittleren Phase
> (Tal bei step 8000: 0.70) und kehrt dann zu ihr zurück.

Ehrliche Vorsicht: 12 Tiefenpunkte (Pythia) machen Formkorrelationen glatter
als bei 24/32; die Diskrepanz Random-Qwen (0.24) vs. Pythia-Init (0.95) kann
Architektur (paralleles Attention+MLP), Init-Schema oder Layerzahl sein —
der nächste billige Test wäre Random-SmolLM2 als dritter Init-Punkt.

## Befund 2 · Die Reihenfolge der Ereignisse (das eigentliche Ziel der Studie)

1. **Rogue-Dims entstehen FRÜH und explosiv:** 0 bis step 1000, **19 bei
   step 4000**, dann Rückbau auf ~6. Ein frühes Übergangsereignis mit
   anschließendem Pruning — nicht graduell.
2. **Die Krümmungsstruktur entsteht SPÄT und abrupt:** Mitte des Trainings
   ist das Krümmungsprofil sogar ANTI-korreliert zum finalen (−0.38 bei
   16k–32k); zwischen 64k und 128k springt es auf +0.99. Das
   „korrigierend→zielgerichtet"-Regime ist eine späte Reorganisation.
3. **Die Kompression kommt ZULETZT und massiv:** eff. Rang ~8 bis step 8000,
   dann 5.4 → 5.1 → **1.8** zwischen 64k und 128k. Der erlernte Fahrplan
   kollabiert am Ende auf fast EINE dominante Richtung pro Tiefe.

**Sequenz: Rogue-Ausbruch (1k–4k) → langes Plateau → gemeinsame späte
Reorganisation (Krümmung + Kompression, 64k–128k), während die Grobform
den Attraktor umkreist.** Krümmung und Kompression springen im selben
Fenster — Kandidat für EIN spätes Reorganisations-Ereignis.

## NACHTRAG · Fein-Sweep 64k–128k (4000er-Schritte, `pythia_fein.json`)

**Antwort auf die Schärfe-Frage: kein scharfer Phasenübergang — aber auch
kein strukturloser Drift.** Das Fenster enthält eine ~40k-Schritte-
**Reorganisationsrampe mit einem lokalisierten Einrast-Ereignis bei ~84k:**

| Messgröße | Verlauf | Charakteristik |
|---|---|---|
| Krümmung r(final) | −0.11 (64k) → 0 (68k) → +0.45 (84k) → +0.66 (100k) → +0.98 (112k) | **graduelle Rampe**, Mittelpunkt ≈ 92k, 10–90 %-Breite ≈ 44k Schritte |
| Form r(trio) | 0.891 (80k) → **0.940 (84k)** → flach | **Sprung +0.05 in EINEM 4k-Schritt** (5× größer als alle Nachbarschritte) — Einrasten auf den Attraktor |
| eff. Rang μ | 5.39 (80k) → **4.23 (84k)** → 3.65 → … → 1.80 (128k) | **Kompressions-BEGINN exakt bei 84k** (größter Einzelschritt −1.2), danach stetig bis Trainingsende — kein Plateau |
| Rogue-Dims | 10 → 12 (84–88k) → **6 (92k)** | Umbau/Pruning zeitgleich mit dem Einrasten |

**Lesart:** Bei ~84k rastet die Fahrplanform auf den modellübergreifenden
Attraktor ein, GLEICHZEITIG beginnt die Rang-Kompression und die Rogue-Menge
wird umgebaut — ein lokalisiertes Ereignis (auf ≤4k Schritte genau). Die
Krümmungs-Reorganisation ist dagegen die breite Hüllkurve darum (68k–112k).
Die Kompression läuft nach dem Einrasten kontinuierlich weiter bis zum
Trainingsende — sie wirkt wie die FOLGE des Einrastens, nicht wie ein
eigenes Ereignis.

## NACHTRAG 2 · LR-Schedule-Abgleich: Das 84k-Einrasten ist KEIN Schedule-Ereignis

Verifizierte Trainingsparameter (EleutherAI-Config `pythia-160m.yml`):
max LR 6.0e-4, min 6.0e-5 (0.1×), linearer Warmup 1 % (1430 Steps),
Cosine-Decay über 143000 Steps, Batch 1024×2048 (≈2.1M Tokens/Step).

| Ereignis (Step) | LR | %max | lokale Steigung |
|---|---|---|---|
| Rogue-Ausbruch 2k–4k | 6.0e-4 | **100 %** | Warmup endet 1430 — Ausbruch beginnt DIREKT in der Max-LR-Phase |
| Krümmungs-Rampe kreuzt 0 (68k) | 3.6e-4 | 59 % | glatt |
| **LOCK-IN (84k)** | 2.6e-4 | **43 %** | **−9.6 %/kStep — identisch zu 80k (−9.8) und 92k (−9.0): völlig glatt** |
| Krümmung fertig (112k) | 1.2e-4 | 20 % | glatt |

Landmarken des Schedules: 50 %-Kreuzung bei ~77.2k, 25 % bei ~105k — **keine
liegt bei 84k**. Kumulierte LR-Masse bei 84k: 83.6 %, Token-Fortschritt 176B
von 300B (58.7 %) — ebenfalls keine ausgezeichneten Werte. Die Pile-Daten
sind durchmischt (kein Curriculum, keine Epochengrenze — 1 Epoche gesamt).

> **Verdikt:** Das Einrast-Ereignis bei 84k hat KEIN Gegenstück im
> LR-Schedule — die Cosine-Kurve ist dort merkmallos glatt. Damit ist die
> naheliegendste exogene Erklärung ausgeschlossen: **Das Einrasten ist ein
> endogenes Ereignis der Lerndynamik**, kein Schedule- oder Daten-Artefakt.
> Umgekehrt hat der frühe Rogue-Ausbruch eine plausible exogene Komponente:
> Er beginnt unmittelbar nach Warmup-Ende in der Max-LR-Phase.
> (Grenzen: n=1 Lauf/Seed; ausgeschlossen ist nur der globale Schedule.)

## Einordnung & nächste Härtung

Die Studie leistet, was sie sollte: aus zwei Schnappschüssen (13) wurde ein
Prozess. Offen/zu härten: (a) ein Seed, ein Modellmaßstab (160m) — Pythia-410m
und/oder zweiter Seed derselben Suite; (b) feinere Checkpoints im
Reorganisations-Fenster 64k–128k (Pythia bietet 1000er-Schritte!) — liegt
dort ein scharfer Übergang (Phasenübergang?) oder ein Drift; (c) Random-Init-
Vergleich dritter Architekturen für die Attraktor-These.
