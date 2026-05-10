#!/usr/bin/env python3
import argparse
import glob
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect merge baseline result summaries.")
    parser.add_argument("--output", default="merge_results", help="Output folder for index files.")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "merge_results/save_s1_*",
            "merge_results/final_s1_*",
            "merge_results/save_s2_*",
            "merge_results/final_s2_*",
            "merge_results/merge_medium_nodef_*",
            "merge_results/fid1000_*",
            "merge_results/celeba_hq_*",
            "merge_results/badmerge_*",
            "save_s1_*",
            "final_s1_*",
            "save_s2_*",
            "final_s2_*",
            "merge_medium_nodef_*",
            "fid1000_*",
            "celeba_hq_*",
            "badmerge_*",
        ],
    )
    parser.add_argument("--copy", action="store_true", help="Copy result folders into the output folder.")
    return parser.parse_args()


def read_summary(result_dir: Path):
    direct = result_dir / "merge_summary.json"
    if direct.exists():
        summaries = [direct]
    else:
        summaries = sorted(result_dir.glob("*/merge_summary.json"))
    if not summaries:
        return None, None
    with summaries[0].open() as f:
        data = json.load(f)
    return summaries[0], data[0] if data else {}


def infer_scenario(name: str):
    if name.startswith("merge_medium_nodef"):
        return "nodef"
    if name.startswith("fid1000_nodef"):
        return "nodef_fid1000"
    if name.startswith("celeba_hq_nodef"):
        return "celeba_hq_nodef"
    if name.startswith("celeba_hq_s1"):
        return "celeba_hq_s1"
    if name.startswith("badmerge_"):
        return "badmerge_" + infer_scenario(name[len("badmerge_"):])
    if "_s1_hat_" in name:
        return "s1_hat"
    if "_s1_cat_" in name:
        return "s1_cat"
    if "_s2_hat_" in name:
        return "s2_hat"
    if "_s2_cat_" in name:
        return "s2_cat"
    return "unknown"


def infer_method(name: str):
    if name.startswith("merge_medium_nodef") or name.startswith("fid1000_nodef") or name.startswith("celeba_hq_nodef"):
        return "no_defense"
    for method in ["diffusion_soup", "clean_finetune", "maxfusion", "dmm", "anp"]:
        if name.endswith(method) or f"_{method}_" in name:
            return method
    return "unknown"


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    result_dirs = []
    for pattern in args.patterns:
        result_dirs.extend(Path(p) for p in glob.glob(pattern))

    for result_dir in sorted(set(result_dirs)):
        if not result_dir.is_dir():
            continue
        summary_path, item = read_summary(result_dir)
        if item is None:
            continue
        model_dirs = sorted(result_dir.glob("*/alpha*/merged_model"))
        row = {
            "name": result_dir.name,
            "path": str(result_dir),
            "scenario": infer_scenario(result_dir.name),
            "method": infer_method(result_dir.name),
            "summary_path": str(summary_path),
            "model_dir": str(model_dirs[0]) if model_dirs else None,
            "alpha": item.get("alpha"),
            "weights": item.get("weights"),
            "fid": item.get("fid"),
            "mse": item.get("mse"),
            "ssim": item.get("ssim"),
            "asr": item.get("asr"),
            "model_save_format": item.get("model_save_format"),
        }
        rows.append(row)
        if args.copy:
            dest = out / result_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(result_dir, dest)

    with (out / "index.json").open("w") as f:
        json.dump(rows, f, indent=2)

    with (out / "README.md").open("w") as f:
        f.write("# Merge Result Index\n\n")
        f.write("This folder indexes completed merge-defense outputs.\n\n")
        f.write("| name | scenario | method | FID | ASR | MSE | SSIM | model save |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            fid = "" if row["fid"] is None else f"{row['fid']:.6f}"
            f.write(
                f"| {row['name']} | {row['scenario']} | {row['method']} | "
                f"{fid} | "
                f"{row['asr']:.6f} | {row['mse']:.6f} | {row['ssim']:.6f} | "
                f"{row['model_save_format']} |\n"
            )

    print(f"Wrote {len(rows)} result entries to {out}")


if __name__ == "__main__":
    main()
