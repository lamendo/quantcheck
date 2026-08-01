"""Studie 36: OLMoE-Checkpoint-Sweep — Experten-Emergenz durch unsere Instrumente.

Praeregistrierung: docs/research/representation_observatory/36_MOE_PRAEREG.md
K-M1 Router-Lock (<=40% Tokens, corr>=0.9 zu final, ab dann dauerhaft)
K-M2 Spearman(tokens, rang) <= -0.7 | K-M3 Spearman(rang, d4) <= -0.7
K-M4 r(final-Profil, Pythia-final-Profil) >= 0.9
bf16, CPU-only, resume-faehig (moe_partial.jsonl).
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_lockin import probe_texts, nll_on, rtn_quantize_, eff_rang  # noqa: E402

ART = HERE.parent / "artifacts" / "topo_steering"
REPO = "allenai/OLMoE-1B-7B-0924"
CKPTS = [
    ("step5000-tokens20B", 20),
    # step180000-tokens754B: Snapshot korrupt + Datei-Lock (Access Violation
    # reproduzierbar, 5 Versuche) -> deklarierter Ersatz durch Nachbarn:
    ("step175000-tokens734B", 734),
    ("step350000-tokens1468B", 1468), ("step525000-tokens2202B", 2202),
    ("step700000-tokens2936B", 2936), ("step875000-tokens3670B", 3670),
    ("step1045000-tokens4383B", 4383), ("step1220000-tokens5117B", 5117),
]
N_PROFILE = 25


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fahrplan_profil(model, tok, texts, n_texts=8):
    """25-Punkte-Profil der Tiefen-Update-Normen (Serie-Definition, normiert)."""
    sums, cnt = None, 0
    for tx in texts[:n_texts]:
        ids = tok(tx, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        h = torch.stack([x[0] for x in out.hidden_states]).float().numpy()
        d = np.diff(h[:, 1:, :], axis=0)
        s = d.sum(axis=1)
        sums = s if sums is None else sums + s
        cnt += d.shape[1]
    mu = sums / cnt                                   # (L, d)
    norms = np.linalg.norm(mu, axis=1)
    xi = np.linspace(0, 1, N_PROFILE)
    x0 = np.linspace(0, 1, len(norms))
    prof = np.interp(xi, x0, norms)
    return (prof / (prof.max() + 1e-12)).round(4).tolist()


def router_nutzung(model, tok, texts, n_texts=8):
    """Mittlere Experten-Nutzungsverteilung je Schicht + Entropie."""
    agg = None
    for tx in texts[:n_texts]:
        ids = tok(tx, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, output_router_logits=True)
        probs = [torch.softmax(r.float(), dim=-1).mean(0) for r in out.router_logits]
        v = torch.stack(probs)                        # (L, n_experts)
        agg = v if agg is None else agg + v
    U = (agg / n_texts).numpy()
    ent = float(np.mean([-(p * np.log(p + 1e-12)).sum() for p in U]))
    return U.round(5).tolist(), round(ent, 4)


def main():
    torch.set_num_threads(max(4, torch.get_num_threads() - 2))
    texts = probe_texts()
    part = ART / "moe_partial.jsonl"
    rows = ([json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
            if part.exists() else [])
    done = {r["tokens_b"] for r in rows}
    tok = AutoTokenizer.from_pretrained(REPO)
    for rev, tokens in CKPTS:
        if tokens in done:
            print(f"skip {tokens}B (bereits gemessen)", flush=True)
            continue
        print(f"lade {rev} ...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            REPO, revision=rev, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        nll0 = nll_on(base, tok, texts, "cpu")
        rang = eff_rang(base, tok, texts, "cpu")
        prof = fahrplan_profil(base, tok, texts)
        usage, ent = router_nutzung(base, tok, texts)
        del base; gc.collect()
        m = AutoModelForCausalLM.from_pretrained(
            REPO, revision=rev, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        rtn_quantize_(m, 4)
        d4 = nll_on(m, tok, texts, "cpu") - nll0
        del m; gc.collect()
        rows.append({"tokens_b": tokens, "nll_bf16": round(nll0, 4),
                     "eff_rang": round(rang, 2), "delta_int4": round(d4, 4),
                     "routing_entropie": ent, "profil": prof, "nutzung": usage})
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        print({k: v for k, v in rows[-1].items() if k not in ("profil", "nutzung")}, flush=True)

    rows.sort(key=lambda r: r["tokens_b"])
    t = [r["tokens_b"] for r in rows]
    rg = [r["eff_rang"] for r in rows]
    d4 = [r["delta_int4"] for r in rows]
    U = [np.asarray(r["nutzung"]).ravel() for r in rows]
    corr_final = [float(np.corrcoef(u, U[-1])[0, 1]) for u in U]
    c_star = None
    for i in range(len(rows)):
        if all(c >= 0.9 for c in corr_final[i:]):
            c_star = t[i]; break
    # K-M4: Pythia-Final-Profil aus dem Telemetrie-Buendel
    B = json.loads((HERE.parent / "docs" / "demos" / "llm_telemetrie_bundle.json")
                   .read_text(encoding="utf-8"))
    pyth = np.asarray(B["training_pythia160m"][-1]["mag_profile"])
    r_attr = float(np.corrcoef(np.asarray(rows[-1]["profil"]), pyth)[0, 1])
    k1 = bool(c_star is not None and c_star <= 0.4 * t[-1])
    k2v, k3v = spearman(t, rg), spearman(rg, d4)
    out = {"repo": REPO,
           "checkpoints": [{k: v for k, v in r.items() if k != "nutzung"} for r in rows],
           "router_corr_zu_final": [round(c, 3) for c in corr_final],
           "router_lock_ab_tokens_b": c_star,
           "K_M1_router_lock_frueh": k1,
           "K_M2_tokens_rang": round(k2v, 3), "K_M2_pass": bool(k2v <= -0.7),
           "K_M3_rang_schaden": round(k3v, 3), "K_M3_pass": bool(k3v <= -0.7),
           "K_M4_attraktor_r": round(r_attr, 3), "K_M4_pass": bool(r_attr >= 0.9)}
    (ART / "moe_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
