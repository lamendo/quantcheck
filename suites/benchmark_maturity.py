"""V2-Kerntest — Sagt das Lock-in-Timing die Benchmark-Reife voraus?

Daten: öffentliche per-Checkpoint-Zero-Shot-Evals (EleutherAI/pythia GitHub)
für pythia-160m, -160m-deduped, -410m. Unsere Emergenz-Kurven aus 14/20.

Vorregistrierte Fragen (VOR Ansicht der Benchmark-Zahlen):
  T1  Reife-Punkt je Task = Step, an dem acc erstmals 90 % der Spanne
      (final − init) erreicht. Liegt der Median der Reife-Punkte im
      Lock-in-Fenster (160m fein: 80k–92k)?
  T2  Beschleunigt/knickt die Benchmark-Kurve im Lock-in-Fenster?
      (Segment-Steigung 73k–93k vs. Nachbarsegmente)
  T3  Cross-Run: Ordnen sich 160m / deduped / 410m konsistent?
Ehrlichkeit: n klein, erster Test der V2-These — Hinweis-Niveau, kein Beweis.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

ART = Path(r"C:\mentis_ai\artifacts\topo_steering")
RAW = "https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1"
MODELS = {"160m": "pythia-160m", "160m-deduped": "pythia-160m-deduped",
          "410m": "pythia-410m"}
TASKS = ["lambada_openai", "piqa", "winogrande", "arc_easy", "sciq", "logiqa"]
CACHE = ART / "pythia_evals"


def fetch(model_dir, fname):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{model_dir}_{fname}"
    if not p.exists():
        url = f"{RAW}/{model_dir}/zero-shot/{fname}"
        try:
            urllib.request.urlretrieve(url, p)
        except Exception:
            return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_steps(model_dir, prefix):
    import urllib.request as ur
    url = f"https://api.github.com/repos/EleutherAI/pythia/contents/evals/pythia-v1/{model_dir}/zero-shot"
    with ur.urlopen(url) as r:
        files = json.load(r)
    steps = {}
    for f in files:
        name = f["name"]
        if name.endswith(".json") and "step" in name:
            s = int(name.split("step")[-1].split(".")[0])
            steps[s] = name
    return dict(sorted(steps.items()))


def curves(model_key):
    mdir = MODELS[model_key]
    steps = list_steps(mdir, model_key)
    out = {t: {} for t in TASKS}
    for s, fname in steps.items():
        d = fetch(mdir, fname)
        if d is None:
            continue
        res = d.get("results", d)
        for t in TASKS:
            if t in res:
                node = res[t]
                acc = node.get("acc", node.get("acc,none"))
                if acc is not None:
                    out[t][s] = float(acc)
    return out


def reife_punkt(steps, accs, frac=0.9):
    a = np.asarray(accs, float)
    lo, hi = a[0], a[-1]
    if hi - lo < 0.02:
        return None
    ziel = lo + frac * (hi - lo)
    for s, v in zip(steps, a):
        if v >= ziel:
            return s
    return None


def main():
    out = {}
    for mk in MODELS:
        cv = curves(mk)
        model_res = {"reife_punkte": {}, "kurven_stuetzstellen": {}}
        for t in TASKS:
            if len(cv[t]) < 8:
                continue
            ss = sorted(cv[t]); aa = [cv[t][s] for s in ss]
            rp = reife_punkt(ss, aa)
            model_res["reife_punkte"][t] = rp
            model_res["kurven_stuetzstellen"][t] = {str(s): round(a, 4)
                                                    for s, a in zip(ss, aa)}
        rps = [v for v in model_res["reife_punkte"].values() if v]
        model_res["median_reife_punkt"] = int(np.median(rps)) if rps else None
        out[mk] = model_res
        print(mk, "Reife-Punkte:", model_res["reife_punkte"],
              "| Median:", model_res["median_reife_punkt"])
    # T2: Segmentsteigungen 160m um das Lock-in-Fenster (Lambada als stärkste Kurve)
    cv160 = out["160m"]["kurven_stuetzstellen"].get("lambada_openai", {})
    ss = sorted(int(s) for s in cv160)
    aa = {int(s): v for s, v in cv160.items()}
    seg = {}
    for lo, hi in [(53000, 73000), (73000, 93000), (93000, 113000), (113000, 133000)]:
        pts = [s for s in ss if lo <= s <= hi]
        if len(pts) >= 2:
            seg[f"{lo//1000}k-{hi//1000}k"] = round(
                (aa[pts[-1]] - aa[pts[0]]) / (pts[-1] - pts[0]) * 1e5, 3)
    out["T2_lambada_segment_slope_160m(acc/100k)"] = seg
    (ART / "v2_kerntest_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k.startswith("T2")}, indent=1))


if __name__ == "__main__":
    main()
