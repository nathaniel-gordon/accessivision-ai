from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def clip_image_similarity(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    model: Optional[torch.nn.Module] = None,
    processor=None,
) -> float:
    if model is None or processor is None:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = next(model.parameters()).device

    def _embed(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        emb = model.get_image_features(pixel_values=x)
        return F.normalize(emb, dim=-1)

    ea = _embed(images_a)
    eb = _embed(images_b)
    return (ea * eb).sum(dim=-1).mean().item()


def _inception_features(images: torch.Tensor) -> torch.Tensor:
    from torchvision.models import inception_v3, Inception_V3_Weights

    net = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).eval()
    net.fc = torch.nn.Identity()
    with torch.no_grad():
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=299, mode="bilinear", align_corners=False)
        feats = net(images)
    return feats


def fid_score(real: torch.Tensor, gen: torch.Tensor) -> float:
    from scipy.linalg import sqrtm

    fr = _inception_features(real).cpu().numpy()
    fg = _inception_features(gen).cpu().numpy()
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False)
    cg = np.cov(fg, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(cr @ cg, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cr + cg - 2 * covmean))


@torch.no_grad()
def bleurt_score(references: List[str], candidates: List[str], model_name: str = "Elron/bleurt-base-512") -> float:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    enc = tok(references, candidates, padding=True, truncation=True, return_tensors="pt")
    return m(**enc).logits.squeeze(-1).mean().item()
