"""Paired comparison of two systems on the same targets, with a bootstrap 95% interval.

delta = A - B per target; the interval comes from 10,000 resamples of the targets (seed 0).
It reflects which targets happen to be in the set. It does not reflect retraining noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .score import BUCKETS


def compare(scores_a: Path, scores_b: Path, metric: str = "tm", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = pd.read_csv(scores_a).merge(pd.read_csv(scores_b)[["id", metric]], on="id", suffixes=("", "_b"))
    d["delta"] = d[metric] - d[metric + "_b"]
    groups = {"overall": d, "family": d[d.family], "no_family": d[~d.family],
              **{f"len{b}": d[d.len_bucket == b] for b in BUCKETS}}
    out = {}
    for k, g in groups.items():
        if len(g) < 5:
            continue
        x = g.delta.to_numpy()
        lo, hi = np.percentile(rng.choice(x, size=(10000, len(x))).mean(1), [2.5, 97.5])
        out[k] = {"n": len(x), "a": round(float(g[metric].mean()), 4),
                  "b": round(float(g[metric + "_b"].mean()), 4), "delta": round(float(x.mean()), 4),
                  "ci": [round(float(lo), 4), round(float(hi), 4)],
                  "win": round(float((x > 0).mean()), 3), "significant": bool(lo > 0 or hi < 0)}
    return out
