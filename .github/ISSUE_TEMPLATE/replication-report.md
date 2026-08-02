---
name: Replication report
about: You ran quantcheck on a checkpoint suite — post the result here
title: "[replication] <model>"
labels: replication
---

Paste the output of:

    python quantcheck.py --model <your-model> --auto-revisions 8 --issue-text

(or attach the report JSON from results/). Please include anything unusual
about the suite: training recipe, token budget, restarts, quantization you
actually deploy with.
