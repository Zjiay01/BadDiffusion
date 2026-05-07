#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Write an index for defended baseline outputs.")
    parser.add_argument("--root", default="defended_models", help="Folder containing copied merge outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for path in sorted(glob.glob(str(root / "merge_medium_*"))):
        path_obj = Path(path)
        summary = path_obj / "merge_summary.json"
        if not summary.exists():
            continue
        with summary.open() as f:
            data = json.load(f)
        item = data[0] if data else {}
        rows.append(
            {
                "name": path_obj.name,
                "path": str(path_obj),
                "alpha": item.get("alpha"),
                "weights": item.get("weights"),
                "fid": item.get("fid"),
                "mse": item.get("mse"),
                "ssim": item.get("ssim"),
                "asr": item.get("asr"),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    with (root / "index.json").open("w") as f:
        json.dump(rows, f, indent=2)

    with (root / "README.md").open("w") as f:
        f.write("# Defended Baseline Outputs\n\n")
        f.write(
            "This folder is a flat copy of completed medium baseline outputs. "
            "The original merge_medium_* directories remain in the project root.\n\n"
        )
        f.write(
            "Note: merge.py saves reusable Diffusers checkpoints only when runs include "
            "--save_model. Older outputs may contain evaluation artifacts only.\n\n"
        )
        f.write("| directory | ASR | MSE | SSIM | FID |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            fid = "N/A" if row["fid"] is None else f"{row['fid']:.6f}"
            f.write(
                f"| {row['name']} | {row['asr']:.6f} | {row['mse']:.6f} | "
                f"{row['ssim']:.6f} | {fid} |\n"
            )

    print(f"Wrote {len(rows)} defended output entries under {root}")


if __name__ == "__main__":
    main()
