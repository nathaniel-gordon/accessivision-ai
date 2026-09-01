from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.logging import JSONLLogger
from .losses import FairnessRegulariser


def _warmup_cosine(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_fn)


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


@dataclass
class TrainingOutput:
    global_step: int
    best_metric: float


class DiffusionTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, cfg: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = AdamW(trainable, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        steps_per_epoch = max(1, len(train_loader) // cfg["training"]["grad_accum_steps"])
        total = steps_per_epoch * cfg["training"]["epochs"]
        self.scheduler = _warmup_cosine(self.optim, cfg["training"]["warmup_steps"], total)

        self.fairness = None
        if cfg.get("fairness", {}).get("enable"):
            self.fairness = FairnessRegulariser(
                attributes=cfg["fairness"]["attributes"],
                ema_window=cfg["fairness"].get("ema_window", 512),
                weight=cfg["fairness"]["weight"],
            )

        self.ema = _EMA(self.model, cfg["training"]["ema_decay"]) if cfg["training"].get("ema_decay") else None
        self.logger = JSONLLogger(Path(cfg["training"]["output_dir"]) / "train.jsonl")
        self.amp_dtype = torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else torch.float16
        self.output_dir = Path(cfg["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best = float("inf")

    def _run_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        pixel = batch["pixel_values"].to(self.device, non_blocking=True)
        variant_id = batch["variant_id"].to(self.device, non_blocking=True)
        prompts = batch["prompt"]
        attrs = batch["attributes"]

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            per_sample = self.model(pixel, prompts, variant_id)
            recon = per_sample.mean()
            loss = recon
            fair = torch.zeros((), device=self.device)
            if self.fairness is not None:
                fair = self.fairness.update_and_penalty(per_sample, attrs)
                loss = loss + fair

        (loss / self.cfg["training"]["grad_accum_steps"]).backward()
        return {"loss": loss.item(), "recon": recon.item(), "fair": fair.item()}

    def train(self) -> TrainingOutput:
        accum = self.cfg["training"]["grad_accum_steps"]
        log_every = self.cfg["logging"]["log_every"]
        save_every = self.cfg["logging"]["save_every"]
        val_every = self.cfg["logging"]["val_every"]

        for epoch in range(self.cfg["training"]["epochs"]):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}")
            for i, batch in enumerate(pbar):
                metrics = self._run_step(batch)
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)
                    self.global_step += 1

                    if self.global_step % log_every == 0:
                        pbar.set_postfix(**metrics)
                        self.logger.log({"step": self.global_step, **metrics,
                                         "lr": self.scheduler.get_last_lr()[0]})
                    if self.global_step % val_every == 0:
                        vm = self.validate()
                        self.logger.log({"step": self.global_step, "val": vm})
                        if vm.get("val_loss", float("inf")) < self.best:
                            self.best = vm["val_loss"]
                            self.save("best.pt")
                    if self.global_step % save_every == 0:
                        self.save("last.pt")
        self.save("last.pt")
        return TrainingOutput(global_step=self.global_step, best_metric=self.best)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            pixel = batch["pixel_values"].to(self.device)
            variant_id = batch["variant_id"].to(self.device)
            per_sample = self.model(pixel, batch["prompt"], variant_id)
            total += per_sample.mean().item() * pixel.size(0)
            n += pixel.size(0)
        out = {"val_loss": total / max(1, n)}
        if self.fairness is not None:
            out.update({f"disp_{k}": v for k, v in self.fairness.disparity().items()})
        return out

    def save(self, name: str) -> None:
        path = self.output_dir / name
        ckpt = {"state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "step": self.global_step, "config": self.cfg}
        if self.ema is not None:
            ckpt["ema"] = {k: v.cpu() for k, v in self.ema.shadow.items()}
        torch.save(ckpt, path)


class LLMTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, cfg: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = AdamW(trainable, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        steps_per_epoch = max(1, len(train_loader) // cfg["training"]["grad_accum_steps"])
        total = steps_per_epoch * cfg["training"]["epochs"]
        warmup = int(cfg["training"]["warmup_ratio"] * total)
        self.scheduler = _warmup_cosine(self.optim, warmup, total)

        self.fairness = None
        if cfg.get("fairness", {}).get("enable"):
            self.fairness = FairnessRegulariser(
                attributes=cfg["fairness"]["attributes"],
                weight=cfg["fairness"]["weight"],
            )

        self.logger = JSONLLogger(Path(cfg["training"]["output_dir"]) / "train.jsonl")
        self.amp_dtype = torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else torch.float16
        self.output_dir = Path(cfg["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best = float("inf")

    def _per_sample_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        bsz, seq, vocab = shift_logits.shape
        ce = F.cross_entropy(
            shift_logits.view(-1, vocab), shift_labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(bsz, seq)
        mask = (shift_labels != -100).float()
        return (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _run_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        pixel = batch["pixel_values"].to(self.device)
        attrs = batch["attributes"]

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            out = self.model(input_ids=input_ids, pixel_values=pixel, attention_mask=attn, labels=labels)
            loss = out.loss
            fair = torch.zeros((), device=self.device)
            if self.fairness is not None and hasattr(out, "logits"):
                per = self._per_sample_loss(out.logits, labels)
                fair = self.fairness.update_and_penalty(per, attrs)
                loss = loss + fair

        (loss / self.cfg["training"]["grad_accum_steps"]).backward()
        return {"loss": loss.item(), "ce": out.loss.item(), "fair": float(fair)}

    def train(self) -> TrainingOutput:
        accum = self.cfg["training"]["grad_accum_steps"]
        log_every = self.cfg["logging"]["log_every"]
        save_every = self.cfg["logging"]["save_every"]
        val_every = self.cfg["logging"]["val_every"]

        for epoch in range(self.cfg["training"]["epochs"]):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}")
            for i, batch in enumerate(pbar):
                metrics = self._run_step(batch)
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % log_every == 0:
                        pbar.set_postfix(**metrics)
                        self.logger.log({"step": self.global_step, **metrics,
                                         "lr": self.scheduler.get_last_lr()[0]})
                    if self.global_step % val_every == 0:
                        vm = self.validate()
                        self.logger.log({"step": self.global_step, "val": vm})
                        if vm["val_loss"] < self.best:
                            self.best = vm["val_loss"]
                            self.save("best.pt")
                    if self.global_step % save_every == 0:
                        self.save("last.pt")
        self.save("last.pt")
        return TrainingOutput(global_step=self.global_step, best_metric=self.best)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                pixel_values=batch["pixel_values"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            total += out.loss.item() * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
        return {"val_loss": total / max(1, n)}

    def save(self, name: str) -> None:
        path = self.output_dir / name
        state = {k: v.cpu() for k, v in self.model.state_dict().items() if "lora" in k or "projector" in k}
        torch.save({"state_dict": state, "step": self.global_step, "config": self.cfg}, path)
