"""Entstehungsgeschichte: Wann lernt ein Transformer den Fahrplan?

Pythia-160m (EleutherAI, öffentliche Checkpoints) über 12 Trainingsstände:
step0 (Init) … step143000 (final). Identische 31 Prompt-Texte (Pythia-
Tokenizer, prefill-only, kein Chat-Template — Basismodell).

Messgrößen je Checkpoint:
  form_r_final   Korrelation der ‖μ(t)‖-Form zum FINALEN Checkpoint
  form_r_trio    Korrelation zur Konsens-Form des trainierten Trios
                 (Qwen/SmolLM2/Qwythos) — entsteht die CROSS-FAMILIEN-Form?
  n_rogue        # Dims mit Varianzanteil > 1 % (irgendeine Tiefe)
  eff_rang_mu    effektiver Rang der μ-Matrix (Kompression)
  curv_r_final   Krümmungsprofil-Korrelation zum finalen Checkpoint
  mag_profile    die Form selbst (für den Report)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import atlas                                        # noqa: E402
from fahrplan_gesetz import fahrplan, shared_rogue, form_deskriptoren  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
GRID = np.linspace(0, 1, 25)
STEPS = [0, 128, 512, 1000, 2000, 4000, 8000, 16000, 32000,
         64000, 128000, 143000]
MODEL = "EleutherAI/pythia-160m"


def prompt_texts():
    import qwythos_extract as qx
    catalog = {i["id"]: i for i in json.loads(
        (HERE / "catalog.json").read_text(encoding="utf-8"))}
    texts = [catalog[i]["prompt"] for i in qx.PICK]
    texts += [q for q, _ in qx.REASONING]
    return texts


def trio_consensus_mag():
    runs = {"qwen05b": ART / "run_obs_qwen05b",
            "smollm2": ART / "run_obs_smollm2",
            "qwythos9b": ART / "run_qwythos_v0"}
    mags = []
    for run in runs.values():
        ids = atlas.item_ids(run)
        rogue = shared_rogue(run, ids)
        mu, _ = fahrplan(run, ids, rogue)
        fd, _ = form_deskriptoren(mu, run, ids, rogue)
        mags.append(fd["mag"])
    return np.mean(mags, axis=0)


def measures(hiddens):
    """hiddens: Liste von (D, T, H)-Arrays → Fahrplan-Messgrößen."""
    deltas_sum, cnt = None, 0
    var_sum = None
    for h in hiddens:
        d = np.diff(h[:, 1:, :], axis=0)
        s = d.sum(axis=1)
        deltas_sum = s if deltas_sum is None else deltas_sum + s
        cnt += d.shape[1]
        v = h.var(axis=1)
        var_sum = v if var_sum is None else var_sum + v
    mu = deltas_sum / cnt                                      # (D-1, H)
    share = var_sum / var_sum.sum(axis=1, keepdims=True)
    n_rogue = int(((share > 0.01).any(axis=0)).sum())
    Dm1 = mu.shape[0]
    t = (np.arange(Dm1) + 0.5) / Dm1
    mag = np.linalg.norm(mu, axis=1)
    mag_i = np.interp(GRID, t, mag / (mag.max() + 1e-12))
    sv = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
    eff = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))
    # Krümmungsprofil (Mittel über Token)
    ps, pn = np.zeros(25), np.zeros(25)
    for h in hiddens[::2]:
        d = np.diff(h[:, 1:, :], axis=0)
        num = (d[:-1] * d[1:]).sum(2)
        den = (np.linalg.norm(d[:-1], axis=2) * np.linalg.norm(d[1:], axis=2) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
        Dm2 = ang.shape[0]
        tb = np.clip((((np.arange(Dm2) + 1.0) / (Dm2 + 1)) * 25).astype(int), 0, 24)
        for b in range(25):
            sel = tb == b
            if sel.any():
                ps[b] += float(ang[sel].mean()); pn[b] += 1
    curv = ps / np.maximum(pn, 1)
    m = curv > 0
    curv_i = np.interp(GRID, GRID[m], curv[m])
    return {"mag": mag_i, "curv": curv_i, "n_rogue": n_rogue, "eff_rang": round(eff, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="", help="z.B. 64000:128000:4000 (Referenz 143000 wird angehängt)")
    ap.add_argument("--out", default="pythia_emergence.json")
    ap.add_argument("--hf-model", default="EleutherAI/pythia-160m")
    args = ap.parse_args()
    global MODEL
    MODEL = args.hf_model
    global STEPS
    if args.steps:
        lo, hi, st = map(int, args.steps.split(":"))
        STEPS = list(range(lo, hi + 1, st))
        if STEPS[-1] != 143000:
            STEPS.append(143000)
    texts = prompt_texts()
    trio = trio_consensus_mag()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    per_step = {}
    for step in STEPS:
        rev = f"step{step}"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=rev, dtype=torch.float32).to(dev).eval()
        except Exception as e:
            print(f"step{step}: FEHLT/FEHLER ({type(e).__name__}) — übersprungen", flush=True)
            continue
        hiddens = []
        for tx in texts:
            ids = tok(tx, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hiddens.append(torch.stack([h[0] for h in out.hidden_states])
                           .cpu().float().numpy())
        per_step[step] = measures(hiddens)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"step{step}: rogue={per_step[step]['n_rogue']} "
              f"rang={per_step[step]['eff_rang']}", flush=True)

    steps_ok = [s for s in STEPS if s in per_step]
    final = per_step[steps_ok[-1]]
    def r(a, b):
        return round(float(np.corrcoef(a, b)[0, 1]), 3)
    curves = {"steps": steps_ok,
              "form_r_final": [r(per_step[s]["mag"], final["mag"]) for s in steps_ok],
              "form_r_trio": [r(per_step[s]["mag"], trio) for s in steps_ok],
              "curv_r_final": [r(per_step[s]["curv"], final["curv"]) for s in steps_ok],
              "n_rogue": [per_step[s]["n_rogue"] for s in steps_ok],
              "eff_rang_mu": [per_step[s]["eff_rang"] for s in steps_ok],
              "mag_profiles": {str(s): per_step[s]["mag"].tolist() for s in steps_ok},
              "final_r_trio": r(final["mag"], trio)}
    (ART / args.out).write_text(
        json.dumps(curves, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: curves[k] for k in
                      ("steps", "form_r_final", "form_r_trio", "n_rogue",
                       "eff_rang_mu", "final_r_trio")}, indent=1))


if __name__ == "__main__":
    main()
