from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def disparity_metrics(
    scores: Sequence[float],
    attributes: Sequence[Tuple[str, ...]],
    names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        buckets: Dict[str, List[float]] = defaultdict(list)
        for s, attr in zip(scores, attributes):
            buckets[attr[i]].append(s)
        means = {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= 5}
        if len(means) < 2:
            out[name] = {"mean": 0.0, "max": 0.0, "groups": means}
            continue
        vals = list(means.values())
        out[name] = {
            "mean": float(max(vals) - min(vals)),
            "max": float(np.std(vals)),
            "groups": means,
        }
    return out


def bootstrap_disparity(
    scores: Sequence[float],
    attributes: Sequence[Tuple[str, ...]],
    names: Sequence[str],
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Dict[str, Tuple[float, float]]:
    rng = rng or np.random.default_rng(0)
    scores = np.asarray(scores, dtype=np.float64)
    attributes = list(attributes)
    n = len(scores)
    per_attr_samples: Dict[str, List[float]] = {name: [] for name in names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_scores = scores[idx]
        sample_attrs = [attributes[i] for i in idx]
        d = disparity_metrics(sample_scores, sample_attrs, names)
        for name in names:
            per_attr_samples[name].append(d[name]["mean"])

    ci: Dict[str, Tuple[float, float]] = {}
    for name, vals in per_attr_samples.items():
        arr = np.asarray(vals)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        ci[name] = (lo, hi)
    return ci
