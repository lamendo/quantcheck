# Architektur oder Training? — Die universelle Fahrplanform ist ERLERNT

**Stand: 2026-07-13 · Code `untrained_fahrplan.py` · Ergebnisse
`untrained_results.json` · Design: Qwen2.5-0.5B-Architektur mit
Zufallsgewichten (2 Seeds), Input = exakt dieselben Token-Sequenzen wie der
trainierte Lauf. Vorregistrierte Kriterien: r≥0.9 → Architektur; r<0.5 →
Training.**

## Hauptergebnis: TRAINING — mit einer wichtigen Nuance

Formkorrelationen der Fahrplan-Stärke ‖μ(t)‖:

| Vergleich | r |
|---|---|
| trainiert ↔ trainiert (3 Familien) | **0.978** |
| random Seed 0 ↔ Seed 1 | **0.940** |
| random ↔ trainiert (alle 3) | **0.23–0.26** |

**Das vorregistrierte Kriterium entscheidet klar: r < 0.5 → erlerntes
Organisationsprinzip.** Aber die Nuance macht den Befund erst richtig stark:
Die Zufallsnetze haben eine EIGENE, hochgradig seed-stabile Form (0.94) —
die Architektur liefert also durchaus *eine* Form, nur eben **nicht die**,
auf die drei unabhängig trainierte Modellfamilien konvergieren.
Das Gesetz der Reihe ist damit präzisiert:

> **Die universelle ‖μ(t)‖-Form ist ein konvergentes, erlerntes
> Organisationsprinzip:** Drei verschiedene Architekturen/Trainings-Pipelines
> (Qwen2.5, Llama-artig, Qwen3.5-Hybrid) erlernen nahezu dieselbe
> Tiefen-Allokation der Bewegung (r≈0.98), die die Architektur allein
> nachweislich nicht vorgibt (r≈0.24). Die Erklärungsfrage verschiebt sich
> von der Architektur zum Trainingsziel: Was am LM-Objective erzwingt
> diese Form?

## Drei Nebenbefunde, jeder für sich wertvoll

1. **Rogue-Dims sind Trainings-Artefakte:** 0 bei beiden Zufallsnetzen,
   15 bei allen drei trainierten Modellen. Die massiven Aktivierungen
   entstehen durch Training — die trim/raw-Doppelführung der ganzen Reihe
   war also eine Kontrolle gegen einen *erlernten* Effekt.
2. **Training komprimiert den Fahrplan:** effektiver Rang der μ-Matrix
   fällt von ~22.6 (random) auf 6–10 (trainiert) — der erlernte Fahrplan
   nutzt deutlich weniger Richtungen.
3. **Auch die Krümmungsstruktur ist erlernt:** Das Krümmungsprofil ist bei
   Zufallsnetzen nicht einmal seed-stabil (r=0.07), bei trainierten Modellen
   geteilt (0.68). Das „korrigierend → zielgerichtet"-Muster aus Phase 1/2
   ist kein Architektur-Reflex.

## Deklarationen & Grenzen

„Untrainiert" = Standard-HF-Initialisierung dieser Architektur (deklariert;
andere Init-Schemata könnten andere Random-Formen liefern — die Konvergenz-
Aussage über trainierte Modelle bleibt davon unberührt). Nur eine Architektur
als Random-Basis (Qwen 0.5B); die trainierte Konvergenz ist über drei
Familien belegt. Nächster Härtungs-Schritt (optional): Random-Init von
SmolLM2-Architektur als zweite Basis; Checkpoint-Reihe (Form-Entstehung
über Trainingsschritte) wäre der Königsweg, braucht aber öffentliche
Zwischenstände (Pythia-Suite!) — Kandidat für die nächste Phase.
