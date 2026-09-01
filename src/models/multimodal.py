from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from .diffusion import AccessibilityDiffusion, DiffusionConfig
from .llm import AltTextLLM, LLMConfig


@dataclass
class MultimodalConfig:
    diffusion: DiffusionConfig
    llm: LLMConfig


class MultimodalAccessibilityModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.diffusion = AccessibilityDiffusion(cfg.diffusion)
        self.llm = AltTextLLM(cfg.llm)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prompt_ids: torch.Tensor,
        variant_ids: torch.Tensor,
        variant_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        alt_text = self.llm.generate(prompt_ids, image)
        accessible_image = self.diffusion.sample(variant_prompts, variant_ids)
        return {"alt_text": alt_text, "accessible_image": accessible_image}
