#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate minimal synthetic architecture samples.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output", default="data/synthetic/dataset.jsonl")
    parser.add_argument("--save-png", action="store_true", help="Optional: save PNGs to data/synthetic/png")
    return parser.parse_args()


def build_sample(i: int) -> dict:
    return {
        "id": i,
        "prompt": "Frontend to backend to database",
        "nodes": ["frontend", "backend", "database"],
        "edges": [["frontend", "backend"], ["backend", "database"]],
    }


def main() -> None:
    args = parse_args()
    count = min(args.num_samples, 1000)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps(build_sample(i)) + "\n")

    if args.save_png:
        png_dir = Path("data/synthetic/png")
        png_dir.mkdir(parents=True, exist_ok=True)
        print("PNG generation is intentionally minimal and disabled by default.")

    print(f"Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")
    main()
