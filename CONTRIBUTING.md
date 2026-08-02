# Contributing

The most valuable contribution is a **measurement**:

```bash
pip install -r requirements.txt
python quantcheck.py --model <your-suite> --auto-revisions 8 --issue-text
```

Post the emitted block as a replication-report issue. Negative results
(flat regimes) are as valuable as confirmations — TinyLlama is already one,
and it is in the README.

Also welcome, roughly in order:
- **AWQ/GPTQ support** in `quantcheck.py` (does calibration-based quant
  close the late-checkpoint gap that RTN shows?)
- Finishing the suspended OLMoE suite (`suites/olmoe_suite.py`, resumable;
  needs ~16 GB memory headroom per load)
- Code cleanup PRs (remaining German comments, typing, logging). Numbers
  must not change; `pytest tests/` guards the basics.
- Bigger/other probe corpora with a comparison against the shipped ones.

Ground rules: JSONs in `results/` are as-run artifacts — never edit them,
add new ones. README claims must stay within what measurements show
(PREREG.md documents how verdicts are declared before runs).
