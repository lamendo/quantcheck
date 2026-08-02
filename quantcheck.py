"""quantcheck — run the rank/quant probe on ANY HF checkpoint suite.

One command, any model with checkpoint revisions:

    python quantcheck.py --model EleutherAI/pythia-160m \\
        --revisions step64000,step84000,step108000,step143000

    python quantcheck.py --model allenai/OLMo-2-0425-1B --list-revisions
    python quantcheck.py --model ... --auto-revisions 8

Outputs a standardized report JSON (with environment metadata) and, with
--issue-text, a ready-to-paste markdown block for a replication report:
https://github.com/lamendo/quantcheck/issues

Per revision it measures (label-free, forward passes only):
  nll_fp        NLL on the declared probe corpus (fp32 or bf16)
  eff_rank      effective rank of the mean depth-update matrix
  eff_rank_norm eff_rank / min(n_transitions, hidden)  (cross-model comparable)
  profile       25-point depth profile of the update norms
  delta_intN    RTN quant damage (NLL_q - NLL_fp) for each requested bit width

Regime labels (declared v0 heuristics, need >= 5 points):
  late-cliff  damage rises with training AND last point >= 2x median of first half
  drift       compression + rising damage without the cliff
  flat        damage span < 0.15 nats and no rank trend
  mixed       anything else
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quant_probe import nll_on, rtn_quantize_, probe_texts, probe_texts_en  # noqa: E402

N_PROFILE = 25


def spearman(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except ImportError:
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


def list_revisions(model):
    url = f"https://huggingface.co/api/models/{model}/refs"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    return [b["name"] for b in d.get("branches", []) if b["name"] != "main"]


def sort_key(rev):
    nums = re.findall(r"\d+", rev)
    return int(nums[0]) if nums else 0


def measure_one(model_id, rev, texts, dtype, bits, dev):
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    nll0 = nll_on(base, tok, texts, dev)

    # depth-update matrix from the first 12 probe texts
    sums, cnt = None, 0
    for tx in texts[:12]:
        ids = tok(tx, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = base(**ids, output_hidden_states=True)
        h = torch.stack([x[0] for x in out.hidden_states]).float().cpu().numpy()
        d = np.diff(h[:, 1:, :], axis=0)
        sums = d.sum(axis=1) if sums is None else sums + d.sum(axis=1)
        cnt += d.shape[1]
    mu = sums / cnt
    sv = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
    rank = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))
    norms = np.linalg.norm(mu, axis=1)
    xi = np.linspace(0, 1, N_PROFILE)
    prof = np.interp(xi, np.linspace(0, 1, len(norms)), norms)
    prof = (prof / (prof.max() + 1e-12)).round(4).tolist()
    n_trans, hidden = mu.shape
    del base
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    row = {"revision": rev, "nll_fp": round(nll0, 4),
           "eff_rank": round(rank, 3),
           "eff_rank_norm": round(rank / min(n_trans, hidden), 4),
           "n_transitions": int(n_trans), "profile": prof}
    for b in bits:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
        rtn_quantize_(m, b)
        row[f"delta_int{b}"] = round(nll_on(m, tok, texts, dev) - nll0, 4)
        del m
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    return row


def classify(rows, bit):
    if len(rows) < 5:
        return {"label": "report-only", "reason": f"only {len(rows)} points (need >= 5)"}
    idx = list(range(len(rows)))
    rk = [r["eff_rank"] for r in rows]
    d4 = [r[f"delta_int{bit}"] for r in rows]
    s_rank = spearman(idx, rk)
    s_dmg = spearman(idx, d4)
    s_link = spearman(rk, d4)
    half = max(1, len(d4) // 2)
    cliff = d4[-1] >= 2 * (np.median(d4[:half]) + 1e-9) and d4[-1] > 0.1
    span = max(d4) - min(d4)
    if s_dmg >= 0.7 and cliff:
        label = "late-cliff"
    elif s_dmg >= 0.7 and s_rank <= -0.7:
        label = "drift"
    elif span < 0.15 and abs(s_rank) < 0.7:
        label = "flat"
    else:
        label = "mixed"
    return {"label": label,
            "spearman_idx_rank": round(s_rank, 3),
            "spearman_idx_damage": round(s_dmg, 3),
            "spearman_rank_damage": round(s_link, 3),
            "damage_span": round(span, 4),
            "last_vs_firsthalf_median": round(float(d4[-1] / (np.median(d4[:half]) + 1e-9)), 2)}


def issue_text(report):
    r = report
    lines = [f"## Replication report: `{r['model']}`", "",
             f"- regime label (v0 heuristic): **{r['verdict']['label']}**",
             f"- device/dtype: {r['environment']['device']} / {r['environment']['dtype']}",
             f"- probe: {r['probe']} ({r['n_texts']} texts)", "",
             "| revision | eff_rank | eff_rank_norm | " +
             " | ".join(f"delta_int{b}" for b in r["bits"]) + " |",
             "|---|---|---|" + "---|" * len(r["bits"])]
    for row in r["checkpoints"]:
        lines.append(f"| {row['revision']} | {row['eff_rank']} | {row['eff_rank_norm']} | " +
                     " | ".join(str(row[f"delta_int{b}"]) for b in r["bits"]) + " |")
    stats = {k: v for k, v in r["verdict"].items() if k != "label"}
    lines += ["", f"stats: `{json.dumps(stats)}`", "",
              f"_generated by quantcheck {r['schema_version']}, "
              f"torch {r['environment']['torch']}, transformers {r['environment']['transformers']}_"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Rank/quant probe over any HF checkpoint suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id")
    ap.add_argument("--revisions", default="", help="comma-separated revision names")
    ap.add_argument("--revisions-file", default="", help="file with one revision per line")
    ap.add_argument("--list-revisions", action="store_true")
    ap.add_argument("--auto-revisions", type=int, default=0,
                    help="pick N evenly spaced revisions automatically")
    ap.add_argument("--probe", default="de", help="de | en | path to a JSON list of texts")
    ap.add_argument("--bits", default="4,8")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--issue-text", action="store_true",
                    help="print a ready-to-paste replication-report markdown block")
    args = ap.parse_args()

    if args.list_revisions:
        for r in sorted(list_revisions(args.model), key=sort_key):
            print(r)
        return

    if args.revisions:
        revs = [r.strip() for r in args.revisions.split(",") if r.strip()]
    elif args.revisions_file:
        revs = [l.strip() for l in Path(args.revisions_file).read_text().splitlines() if l.strip()]
    elif args.auto_revisions:
        allr = sorted(list_revisions(args.model), key=sort_key)
        if not allr:
            sys.exit("no revisions found — is this a checkpoint suite?")
        pick = np.linspace(0, len(allr) - 1, args.auto_revisions).round().astype(int)
        revs = [allr[i] for i in sorted(set(pick))]
        print("auto-selected revisions:", ", ".join(revs), flush=True)
    else:
        sys.exit("need --revisions, --revisions-file, --auto-revisions or --list-revisions")

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        try:
            from huggingface_hub import model_info
            n = (model_info(args.model).safetensors or {}).total or 0
        except Exception:
            n = 0
        dtype = torch.float32 if 0 < n < 1.5e9 else (torch.float32 if n == 0 else torch.bfloat16)
    else:
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    bits = [int(b) for b in args.bits.split(",")]
    if args.probe in ("de", "en"):
        texts = probe_texts() if args.probe == "de" else probe_texts_en()
    else:
        texts = json.loads(Path(args.probe).read_text(encoding="utf-8"))

    safe = args.model.replace("/", "_")
    out_path = Path(args.out) if args.out else HERE / "results" / f"quantcheck_{safe}.json"
    part = out_path.with_suffix(".partial.jsonl")
    rows = []
    if part.exists() and not args.no_resume:
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"resume: {len(rows)} revisions already measured", flush=True)
    done = {r["revision"] for r in rows}

    for rev in revs:
        if rev in done:
            continue
        print(f"measuring {rev} ...", flush=True)
        row = measure_one(args.model, rev, texts, dtype, bits, dev)
        rows.append(row)
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(" ", {k: v for k, v in row.items() if k != "profile"}, flush=True)

    order = {rev: i for i, rev in enumerate(revs)}
    rows.sort(key=lambda r: order.get(r["revision"], 1e9))
    report = {
        "schema_version": "quantcheck-report-v1",
        "model": args.model, "revisions": revs, "bits": bits,
        "probe": args.probe, "n_texts": len(texts),
        "checkpoints": rows,
        "verdict": classify(rows, bits[0]),
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": (torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or "cpu"),
            "dtype": str(dtype).replace("torch.", ""),
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nreport -> {out_path}")
    print("verdict:", json.dumps(report["verdict"]))
    if args.issue_text:
        print("\n" + "=" * 60 + "\n" + issue_text(report))


if __name__ == "__main__":
    main()
