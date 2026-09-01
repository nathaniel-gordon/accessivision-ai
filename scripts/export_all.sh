#!/usr/bin/env bash
set -euo pipefail

# Export the latest checkpoints to CoreML
python scripts/export_coreml.py --checkpoint runs/diffusion/last.pt --out runs/diffusion.mlpackage
