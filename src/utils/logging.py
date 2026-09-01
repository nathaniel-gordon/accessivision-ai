from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
