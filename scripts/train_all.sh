#!/usr/bin/env bash
set -euo pipefail

# Train all three models in sequence
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_llm.py       --config configs/llm.yaml
