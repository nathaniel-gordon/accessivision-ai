from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _attr(rng: random.Random) -> dict:
    return {
        "skin_tone": rng.choice(["light", "medium", "dark"]),
        "gender_presentation": rng.choice(["fem", "masc", "androgynous"]),
        "age_bucket": rng.choice(["child", "adult", "senior"]),
    }


def _record(rng: random.Random, i: int) -> dict:
    variants = ["high_contrast", "simplified", "line_art"]
    return {
        "image_path": f"images/{i:06d}.jpg",
        "prompt": f"accessible rendering of scene {i}",
        "variant": rng.choice(variants),
        "attributes": _attr(rng),
        "alt_text": f"A {rng.choice(['bright','detailed','minimal'])} scene showing subject {i}.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data")
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split, n in (("accessibility_train", args.n_train),
                     ("accessibility_val", args.n_val),
                     ("alttext_train", args.n_train),
                     ("alttext_val", args.n_val)):
        path = out / f"{split}.jsonl"
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps(_record(rng, i)) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
