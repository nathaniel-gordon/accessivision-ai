from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class LLMConfig:
    base: str = "microsoft/Phi-3-mini-4k-instruct"
    vision_encoder: str = "openai/clip-vit-large-patch14"
    projector_hidden: int = 2048
    load_in_4bit: bool = True
    qlora_r: int = 32
    qlora_alpha: int = 64
    qlora_dropout: float = 0.05
    qlora_targets: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AltTextLLM(nn.Module):
    IMAGE_TOKEN = "<|image|>"

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel

        quant = None
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
        if self.IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.IMAGE_TOKEN]})

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.base,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.vision = CLIPVisionModel.from_pretrained(cfg.vision_encoder)
        self.vision.requires_grad_(False)

        self.projector = VisionProjector(
            in_dim=self.vision.config.hidden_size,
            hidden=cfg.projector_hidden,
            out_dim=self.llm.config.hidden_size,
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self._inject_qlora()

    def _inject_qlora(self) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.cfg.load_in_4bit:
            self.llm = prepare_model_for_kbit_training(self.llm)

        lcfg = LoraConfig(
            r=self.cfg.qlora_r,
            lora_alpha=self.cfg.qlora_alpha,
            lora_dropout=self.cfg.qlora_dropout,
            target_modules=list(self.cfg.qlora_targets),
            task_type="CAUSAL_LM",
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lcfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values=pixel_values).last_hidden_state
        return self.projector(feats)

    def _fuse(self, input_ids: torch.Tensor, image_feats: torch.Tensor) -> torch.Tensor:
        embed_layer = self.llm.get_input_embeddings()
        text_embeds = embed_layer(input_ids)
        mask = input_ids == self.image_token_id
        bsz, seq, dim = text_embeds.shape
        num_img_tokens = image_feats.shape[1]

        for b in range(bsz):
            idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            start = idx[0].item()
            prefix = text_embeds[b, :start]
            suffix = text_embeds[b, start + 1 :]
            fused = torch.cat([prefix, image_feats[b], suffix], dim=0)[:seq]
            text_embeds[b] = fused
        return text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        image_feats = self.encode_image(pixel_values)
        inputs_embeds = self._fuse(input_ids, image_feats)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        image_feats = self.encode_image(pixel_values)
        inputs_embeds = self._fuse(input_ids, image_feats)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
