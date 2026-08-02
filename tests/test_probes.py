"""Smoke tests — no model downloads, runs in seconds: pytest tests/"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from quant_probe import rtn_quantize_, probe_texts, probe_texts_en  # noqa: E402


def test_probe_corpora_load_and_are_unique():
    de, en = probe_texts(), probe_texts_en()
    assert len(de) == 31 and len(set(de)) == 31
    assert len(en) == 20 and len(set(en)) == 20


def test_rtn_int8_is_near_lossless_int4_is_not():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias=False)
    w0 = lin.weight.data.clone()
    m8 = torch.nn.Linear(64, 64, bias=False); m8.weight.data = w0.clone()
    rtn_quantize_(m8, 8)
    m4 = torch.nn.Linear(64, 64, bias=False); m4.weight.data = w0.clone()
    rtn_quantize_(m4, 4)
    e8 = (m8.weight.data - w0).abs().mean().item()
    e4 = (m4.weight.data - w0).abs().mean().item()
    assert e8 < e4 / 5, (e8, e4)
    # symmetric per-row scale: max magnitude preserved within one step
    assert torch.allclose(m4.weight.data.abs().amax(1), w0.abs().amax(1), rtol=0.2)


def test_effective_rank_orders_low_vs_high_rank_updates():
    rng = np.random.default_rng(0)
    def eff(mu):
        sv = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
        return (sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12)
    low = np.outer(rng.normal(size=24), rng.normal(size=256))
    low += 0.01 * rng.normal(size=low.shape)
    high = rng.normal(size=(24, 256))
    assert eff(low) < 2.0 < eff(high)


def test_trio_and_pythia_reference_forms_ship_correctly():
    trio = json.loads((ROOT / "probes" / "trio_consensus_mag.json").read_text())
    pyth = json.loads((ROOT / "probes" / "pythia160m_final_mag.json").read_text())
    assert len(trio["mag"]) == 25 and len(pyth["mag"]) == 25
    assert max(pyth["mag"]) <= 1.0 + 1e-9
