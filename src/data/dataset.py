from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_image_transform

VARIANT_TO_ID = {"high_contrast": 0, "simplified": 1, "line_art": 2}
ATTRIBUTES = ("skin_tone", "gender_presentation", "age_bucket")


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Record:
    image_path: str
    prompt: str
    variant: str
    attributes: Dict[str, str]
    alt_text: Optional[str] = None


class AccessibilityDataset(Dataset):
    def __init__(self, manifest: str, image_size: int = 512):
        self.root = Path(manifest).parent
        self.items = [Record(**r) for r in _read_jsonl(Path(manifest))]
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.items[idx]
        img = Image.open(self.root / r.image_path).convert("RGB")
        pixel = self.transform(img)
        variant_id = VARIANT_TO_ID[r.variant]
        attr = tuple(r.attributes.get(a, "unknown") for a in ATTRIBUTES)
        return {
            "pixel_values": pixel,
            "prompt": r.prompt,
            "variant_id": torch.tensor(variant_id, dtype=torch.long),
            "attributes": attr,
        }


class AltTextDataset(Dataset):
    def __init__(self, manifest: str, tokenizer, prompt_template: str, image_size: int = 224, max_length: int = 1024):
        self.root = Path(manifest).parent
        self.items = [Record(**r) for r in _read_jsonl(Path(manifest))]
        self.tokenizer = tokenizer
        self.template = prompt_template
        self.transform = build_image_transform(image_size, clip_mean=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def _format(self, instruction: str, answer: str) -> Dict[str, torch.Tensor]:
        prompt = self.template.format(image="<|image|>", instruction=instruction)
        full = prompt + answer + self.tokenizer.eos_token
        enc = self.tokenizer(full, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        prompt_ids = self.tokenizer(prompt, max_length=self.max_length, truncation=True, return_tensors="pt").input_ids[0]
        labels = enc.input_ids[0].clone()
        labels[: len(prompt_ids)] = -100
        labels[enc.attention_mask[0] == 0] = -100
        return {
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.items[idx]
        img = Image.open(self.root / r.image_path).convert("RGB")
        pixel = self.transform(img)
        encoded = self._format(r.prompt, r.alt_text or "")
        attr = tuple(r.attributes.get(a, "unknown") for a in ATTRIBUTES)
        encoded.update({"pixel_values": pixel, "attributes": attr})
        return encoded


def _collate(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def make_dataloader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=shuffle,
    )
