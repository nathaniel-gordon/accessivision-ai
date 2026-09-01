from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


class _UNetWrapper(torch.nn.Module):
    def __init__(self, unet, variant_embed):
        super().__init__()
        self.unet = unet
        self.variant_embed = variant_embed

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        variant_id: torch.Tensor,
    ) -> torch.Tensor:
        var = self.variant_embed(variant_id).unsqueeze(1)
        ctx = encoder_hidden_states + var
        return self.unet(sample, timestep, encoder_hidden_states=ctx).sample


def export_diffusion_coreml(
    model,
    out_path: str,
    latent_shape: Tuple[int, int, int, int] = (1, 4, 64, 64),
    seq_len: int = 77,
    embed_dim: int = 768,
    quantize: str = "int8",
) -> Path:
    import coremltools as ct

    model = model.eval().cpu()
    wrapper = _UNetWrapper(model.unet, model.variant).eval()
    sample = torch.randn(latent_shape)
    timestep = torch.tensor([981], dtype=torch.float32)
    context = torch.randn(latent_shape[0], seq_len, embed_dim)
    variant = torch.zeros(latent_shape[0], dtype=torch.long)

    traced = torch.jit.trace(wrapper, (sample, timestep, context, variant), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape),
            ct.TensorType(name="timestep", shape=timestep.shape),
            ct.TensorType(name="encoder_hidden_states", shape=context.shape),
            ct.TensorType(name="variant_id", shape=variant.shape, dtype=int),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    if quantize == "int8":
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8))
        mlmodel = palettize_weights(mlmodel, cfg)
    elif quantize == "fp16":
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    return out


def export_llm_coreml(projector: torch.nn.Module, out_path: str, in_dim: int = 1024) -> Path:
    import coremltools as ct

    projector = projector.eval().cpu()
    example = torch.randn(1, 257, in_dim)
    traced = torch.jit.trace(projector, example, strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="vision_features", shape=example.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    return out
