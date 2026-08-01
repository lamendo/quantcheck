# GGUF conversion fix for transformers >= 5

`convert_hf_to_gguf.py` (llama.cpp) fails on GPT-NeoX/Pythia configs saved by
transformers 5.x with `KeyError: 'rotary_pct'` — the field was renamed to
`partial_rotary_factor`.

One-line fix in the GPT-NeoX conversion path:

```python
# before
rot_pct = hparams["rotary_pct"]
# after
rot_pct = hparams.get("rotary_pct", hparams.get("partial_rotary_factor", 0.25))
```

Needed only for `gguf_probe.py` runs against checkpoints re-saved with
transformers 5.x. Upstream llama.cpp may have fixed this since.
