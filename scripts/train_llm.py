from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import AltTextDataset, make_dataloader
from models import AltTextLLM
from models.llm import LLMConfig
from training import LLMTrainer
from utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    lcfg = LLMConfig(
        base=cfg["model"]["base"],
        vision_encoder=cfg["model"]["vision_encoder"],
        projector_hidden=cfg["model"]["projector_hidden"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        qlora_r=cfg["model"]["qlora"]["r"],
        qlora_alpha=cfg["model"]["qlora"]["alpha"],
        qlora_dropout=cfg["model"]["qlora"]["dropout"],
        qlora_targets=tuple(cfg["model"]["qlora"]["targets"]),
    )
    model = AltTextLLM(lcfg)

    train_ds = AltTextDataset(
        cfg["data"]["train_manifest"], model.tokenizer, cfg["data"]["prompt_template"],
        max_length=cfg["data"]["max_length"],
    )
    val_ds = AltTextDataset(
        cfg["data"]["val_manifest"], model.tokenizer, cfg["data"]["prompt_template"],
        max_length=cfg["data"]["max_length"],
    )
    train_dl = make_dataloader(train_ds, cfg["training"]["batch_size"], shuffle=True)
    val_dl = make_dataloader(val_ds, cfg["training"]["batch_size"], shuffle=False)

    trainer = LLMTrainer(model, train_dl, val_dl, cfg)
    out = trainer.train()
    print(f"done: step={out.global_step} best={out.best_metric:.4f}")


if __name__ == "__main__":
    main()
