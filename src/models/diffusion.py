from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    base: str = "runwayml/stable-diffusion-v1-5"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    prediction_type: str = "epsilon"
    variants: tuple = ("high_contrast", "simplified", "line_art")


class VariantConditioner(nn.Module):
    def __init__(self, num_variants: int, embed_dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(num_variants, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, variant_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(variant_ids)


class AccessibilityDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.vae = AutoencoderKL.from_pretrained(cfg.base, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.base, subfolder="unet")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.base, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.base, subfolder="tokenizer")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.base, subfolder="scheduler")
        self.variant = VariantConditioner(len(cfg.variants), self.text_encoder.config.hidden_size)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self._inject_lora()

    def _inject_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=list(self.cfg.lora_targets),
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lcfg)

    def encode_prompt(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(
            prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            return self.text_encoder(**tok).last_hidden_state

    def condition(
        self, prompts: List[str], variant_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        text = self.encode_prompt(prompts, device)
        var = self.variant(variant_ids).unsqueeze(1)
        return text + var

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: List[str],
        variant_ids: torch.Tensor,
    ) -> torch.Tensor:
        device = pixel_values.device
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        cond = self.condition(prompts, variant_ids, device)
        pred = self.unet(noisy, timesteps, encoder_hidden_states=cond).sample
        target = noise if self.cfg.prediction_type == "epsilon" else latents
        return F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))

    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        variant_ids: torch.Tensor,
        steps: int = 30,
        guidance: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> torch.Tensor:
        from diffusers import DPMSolverMultistepScheduler

        device = next(self.parameters()).device
        scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(steps, device=device)
        shape = (len(prompts), self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, device=device) * scheduler.init_noise_sigma

        cond = self.condition(prompts, variant_ids, device)
        uncond = self.condition([""] * len(prompts), variant_ids, device)

        for t in scheduler.timesteps:
            lat_in = torch.cat([latents, latents])
            ctx = torch.cat([uncond, cond])
            pred = self.unet(lat_in, t, encoder_hidden_states=ctx).sample
            p_u, p_c = pred.chunk(2)
            pred = p_u + guidance * (p_c - p_u)
            latents = scheduler.step(pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return (image.clamp(-1, 1) + 1) / 2
