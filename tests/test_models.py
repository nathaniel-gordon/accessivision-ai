from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.diffusion import VariantConditioner


def test_variant_conditioner_shape():
    v = VariantConditioner(num_variants=3, embed_dim=64)
    ids = torch.tensor([0, 2, 1])
    out = v(ids)
    assert out.shape == (3, 64)


def test_variant_conditioner_distinct_embeddings():
    v = VariantConditioner(num_variants=3, embed_dim=32)
    ids = torch.arange(3)
    out = v(ids)
    diffs = (out.unsqueeze(0) - out.unsqueeze(1)).norm(dim=-1)
    eye = torch.eye(3).bool()
    assert (diffs[~eye] > 0).all()
