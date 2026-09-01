from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deployment import export_diffusion_coreml, export_llm_coreml
from models import AccessibilityDiffusion, AltTextLLM
from models.diffusion import DiffusionConfig
from models.llm import LLMConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_checkpoint")
    p.add_argument("--llm_checkpoint")
    p.add_argument("--output_dir", default="exports/coreml")
    p.add_argument("--quantize", choices=["fp16", "int8", "none"], default="int8")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.diffusion_checkpoint:
        ckpt = torch.load(args.diffusion_checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
        dcfg = DiffusionConfig(
            base=cfg["model"]["base"],
            variants=tuple(cfg["data"]["variants"]),
        )
        model = AccessibilityDiffusion(dcfg)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        path = export_diffusion_coreml(model, str(out_dir / "AccessibleDiffusion.mlpackage"), quantize=args.quantize)
        print(f"diffusion -> {path}")

    if args.llm_checkpoint:
        ckpt = torch.load(args.llm_checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
        lcfg = LLMConfig(
            base=cfg["model"]["base"],
            vision_encoder=cfg["model"]["vision_encoder"],
            projector_hidden=cfg["model"]["projector_hidden"],
            load_in_4bit=False,
        )
        model = AltTextLLM(lcfg)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        path = export_llm_coreml(
            model.projector, str(out_dir / "AltTextProjector.mlpackage"),
            in_dim=model.vision.config.hidden_size,
        )
        print(f"llm-projector -> {path}")


if __name__ == "__main__":
    main()
