from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import (
    bleurt_score,
    bootstrap_disparity,
    clip_image_similarity,
    disparity_metrics,
    fid_score,
)
from utils import load_config


def _load_eval_records(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    records = _load_eval_records(cfg["dataset"])

    refs = [r["reference_alt_text"] for r in records if "reference_alt_text" in r]
    cands = [r["generated_alt_text"] for r in records if "generated_alt_text" in r]
    attrs = [tuple(r["attributes"].get(a, "unknown") for a in cfg["fairness"]["attributes"]) for r in records]

    report = {"n_samples": len(records)}
    if refs and cands:
        report["bleurt"] = bleurt_score(refs, cands)

    scores = [r.get("bleurt_per_sample", 0.0) for r in records]
    if any(scores):
        report["disparity"] = disparity_metrics(scores, attrs, cfg["fairness"]["attributes"])
        report["disparity_ci"] = bootstrap_disparity(
            scores, attrs, cfg["fairness"]["attributes"],
            n_bootstrap=cfg["fairness"]["bootstraps"],
        )

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
