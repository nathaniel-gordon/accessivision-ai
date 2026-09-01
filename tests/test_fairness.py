from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.fairness import bootstrap_disparity, disparity_metrics
from training.losses import FairnessRegulariser


def test_disparity_metrics_returns_zero_when_balanced():
    scores = [1.0] * 20
    attrs = [("a",)] * 10 + [("b",)] * 10
    out = disparity_metrics(scores, attrs, ("grp",))
    assert out["grp"]["mean"] == pytest.approx(0.0)


def test_disparity_metrics_detects_gap():
    scores = [0.2] * 10 + [0.8] * 10
    attrs = [("a",)] * 10 + [("b",)] * 10
    out = disparity_metrics(scores, attrs, ("grp",))
    assert out["grp"]["mean"] == pytest.approx(0.6, abs=1e-6)


def test_bootstrap_disparity_confidence_interval():
    scores = [0.2] * 50 + [0.8] * 50
    attrs = [("a",)] * 50 + [("b",)] * 50
    ci = bootstrap_disparity(scores, attrs, ("grp",), n_bootstrap=50)
    lo, hi = ci["grp"]
    assert 0.0 <= lo <= hi <= 1.0


def test_fairness_regulariser_updates_buckets():
    reg = FairnessRegulariser(("skin_tone",), ema_window=64, weight=0.5)
    losses = torch.tensor([0.5, 1.5, 0.5, 1.5])
    attrs = [("light",), ("dark",), ("light",), ("dark",)]
    for _ in range(5):
        _ = reg.update_and_penalty(losses, attrs)
    disp = reg.disparity()
    assert disp["skin_tone"] > 0
