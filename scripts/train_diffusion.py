from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import AccessibilityDataset, make_dataloader
from models import AccessibilityDiffusion
from models.diffusion import DiffusionConfig
from training import DiffusionTrainer
from utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    dcfg = DiffusionConfig(
        base=cfg["model"]["base"],
        lora_rank=cfg["model"]["lora_rank"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_targets=tuple(cfg["model"]["lora_targets"]),
        prediction_type=cfg["model"]["prediction_type"],
        variants=tuple(cfg["data"]["variants"]),
    )
    model = AccessibilityDiffusion(dcfg)

    train_ds = AccessibilityDataset(cfg["data"]["train_manifest"], cfg["data"]["image_size"])
    val_ds = AccessibilityDataset(cfg["data"]["val_manifest"], cfg["data"]["image_size"])
    train_dl = make_dataloader(train_ds, cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_dl = make_dataloader(val_ds, cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    trainer = DiffusionTrainer(model, train_dl, val_dl, cfg)
    out = trainer.train()
    print(f"done: step={out.global_step} best={out.best_metric:.4f}")


if __name__ == "__main__":
    main()
