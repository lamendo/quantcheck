# 36 — MoE: Experten-Emergenz durch unsere Instrumente (Präregistrierung)

**Datum:** 2026-07-30 (VOR allen Läufen) · **Skript:** `topo_steering/moe_sweep.py` ·
**Subjekt:** allenai/OLMoE-1B-7B-0924 (64 Experten, 8 aktiv; 245 öffentliche
Checkpoints, 20B–5117B Tokens; gleiche Werkstatt wie das bestätigte OLMo-2).

## Fragen (aus dem Gespräch: „inwiefern korrelieren unsere Untersuchungen
mit der emergenten Natur der Experts?")

- **P1 Router-Lock:** Die Experten-Spezialisierung ist ein frühes,
  endogenes Einrast-Ereignis (MoE-Pendant zum Pythia-Lock-in).
- **P2 Kompression & Fragilität:** Auch der MoE-Reststrom komprimiert über
  das Training, und der Rang↔Int4-Schaden-Link gilt (4. Familie).
- **P4 Attraktor:** Die universelle Fahrplanform gilt auch für MoE
  (Architektur ersetzt nur die MLPs — der Reststrom bleibt) → 8. Lauf.

## Messungen je Checkpoint (8 Checkpoints: 20B / 754B / 1468B / 2202B /
2936B / 3670B / 4383B / 5117B Tokens; bf16, CPU, deklarierte Probe DE)

1. nll_fp (bf16-Basis), eff_rang, delta_int4 (RTN, alle Linear-Schichten)
   — identische Pipeline wie Serie 25/31/32.
2. Fahrplan-Profil (25-Punkte, normiert) — für P4.
3. **Router-Telemetrie** (output_router_logits auf der Probe):
   je Schicht mittlere Experten-Nutzungsverteilung (16×64) + Routing-Entropie.

## Kriterien (bindend)

- **K-M1 (P1):** Es existiert ein Checkpoint c* mit Token-Fortschritt
  ≤ 40 % (≤ 2047B), ab dem corr(Nutzung_c, Nutzung_final) ≥ 0.9 für ALLE
  folgenden Checkpoints gilt → Router rastet früh ein.
  (corr über den konkatenierten 16×64-Nutzungsvektor.)
- **K-M2 (P2a):** Spearman(Tokens, eff_rang) ≤ −0.7 (Kompression;
  OLMo-2-dicht: −0.883).
- **K-M3 (P2b):** Spearman(eff_rang, delta_int4) ≤ −0.7 (Mechanismus-Link;
  bisher −0.96/−0.86/−0.87).
- **K-M4 (P4):** Pearson r(finales OLMoE-Profil, finales Pythia-160m-Profil,
  je 25 Punkte normiert) ≥ 0.9 (Attraktor; bisherige Familienpaare 0.94–1.00).

Jede Hypothese wird UNABHÄNGIG verdiktet (kein Aggregat-PASS); jedes
verfehlte K wird als eigenständiges Negativergebnis dokumentiert.

## Exploratorisch (nur finaler Checkpoint, KEIN Kriterium)

Quant-Attribution: RTN-Int4 nur auf Experten-Gewichte vs nur auf
Attention-Gewichte → wo wohnt der Schaden im MoE?

## Deklarierte Abweichung (2026-07-31, vor Messung des Punkts)

Checkpoint 2 (step180000-tokens754B): Download-Snapshot korrupt (reproduzierbare
Access Violation beim ersten Forward, 5 Versuche) + Datei-Lock durch externen
Prozess, Löschung nicht möglich → ersetzt durch Nachbar-Revision
**step175000-tokens734B** (Δ 20 B Tokens ≈ 0,4 % der Gesamtachse; kein
Einfluss auf Kriterien, da alle K auf Rangkorrelationen/Endpunkt beruhen).

## Deklarierte Grenzen

Eine MoE-Familie; bf16 statt fp32-Basis (7B auf 32-GB-RAM — Abweichung von
der Serie, deklariert; delta_int4 bleibt Differenzmaß auf gleicher Basis);
CPU-Laufzeit mehrere Stunden, download-gebunden (~14 GB × 8 auf D:);
Router-Nutzung auf deklarierter Probe gemessen, nicht auf Trainingsdaten;
n=8 Checkpoints für Spearman-Kriterien (wie TinyLlama-Serie).
