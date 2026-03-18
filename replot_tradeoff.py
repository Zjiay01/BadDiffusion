import argparse
import json
import os
from typing import Dict, List, Optional

# Usage examples:
# 1) Use a directory containing merge_summary.json / merge_summary.txt
#    python replot_tradeoff.py --summary_path ./merge_results_quick
#
# 2) Use a specific summary file
#    python replot_tradeoff.py --summary_path ./merge_results_quick/merge_summary.json
#
# 3) Custom output path
#    python replot_tradeoff.py --summary_path ./merge_results_quick --output_path ./plots/my_tradeoff.png
#
# 4) Custom title and dpi
#    python replot_tradeoff.py --summary_path ./merge_results_quick --title "CIFAR10 Merge Tradeoff" --dpi 200
#
# Notes:
# - --summary_path supports both file path and directory path.
# - If FID is missing (N/A or null), the script still plots ASR and annotates FID unavailability.
# - Default output is <summary_dir>/asr_fid_tradeoff.png.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline redraw for ASR-FID tradeoff plot from merge summary data."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        required=True,
        help=(
            "Path to merge summary JSON/TXT file, or a directory containing "
            "merge_summary.json / merge_summary.txt"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output image path. Default: <summary_dir>/asr_fid_tradeoff.png",
    )
    parser.add_argument("--title", type=str, default="Backdoor Survivability under Model Merging")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def resolve_summary_file(summary_path: str) -> str:
    if os.path.isfile(summary_path):
        return summary_path

    if not os.path.isdir(summary_path):
        raise FileNotFoundError(f"summary path does not exist: {summary_path}")

    json_path = os.path.join(summary_path, "merge_summary.json")
    txt_path = os.path.join(summary_path, "merge_summary.txt")

    if os.path.isfile(json_path):
        return json_path
    if os.path.isfile(txt_path):
        return txt_path

    raise FileNotFoundError(
        "No merge_summary.json or merge_summary.txt found under directory: "
        f"{summary_path}"
    )


def _maybe_float(text: str) -> Optional[float]:
    t = text.strip()
    if t.upper() == "N/A":
        return None
    return float(t)


def load_results_from_txt(txt_file: str) -> List[Dict]:
    results: List[Dict] = []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("alpha") or line.startswith("-"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 5:
                continue

            alpha = float(parts[0])
            fid = _maybe_float(parts[1])
            mse = float(parts[2])
            ssim = float(parts[3])
            asr = float(parts[4])
            results.append({"alpha": alpha, "fid": fid, "mse": mse, "ssim": ssim, "asr": asr})

    if len(results) == 0:
        raise ValueError(f"No valid rows found in summary txt: {txt_file}")

    return results


def load_results(summary_file: str) -> List[Dict]:
    _, ext = os.path.splitext(summary_file.lower())
    if ext == ".json":
        with open(summary_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Invalid or empty summary json: {summary_file}")
        return data

    if ext == ".txt":
        return load_results_from_txt(summary_file)

    raise ValueError(f"Unsupported summary file extension: {summary_file}")


def plot_tradeoff(results: List[Dict], output_path: str, title: str, dpi: int):
    import matplotlib.pyplot as plt

    alphas = [float(r["alpha"]) for r in results]
    asrs = [float(r["asr"]) for r in results]
    fid_pairs = [(float(r["alpha"]), float(r["fid"])) for r in results if r.get("fid") is not None]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_asr = "tab:red"
    color_fid = "tab:blue"

    ax1.set_xlabel("Alpha (backdoor model weight)")
    ax1.set_ylabel("ASR", color=color_asr)
    ax1.plot(alphas, asrs, "o-", color=color_asr, label="ASR")
    ax1.tick_params(axis="y", labelcolor=color_asr)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(fid_pairs) > 0:
        ax2 = ax1.twinx()
        fid_alphas = [p[0] for p in fid_pairs]
        fids = [p[1] for p in fid_pairs]
        ax2.set_ylabel("FID", color=color_fid)
        ax2.plot(fid_alphas, fids, "s--", color=color_fid, label="FID")
        ax2.tick_params(axis="y", labelcolor=color_fid)
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    else:
        ax1.legend(lines1, labels1, loc="center left")
        ax1.text(
            0.98,
            0.02,
            "FID unavailable",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="tab:gray",
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    summary_file = resolve_summary_file(args.summary_path)
    results = load_results(summary_file=summary_file)

    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_dir = os.path.dirname(summary_file)
        output_path = os.path.join(output_dir, "asr_fid_tradeoff.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_tradeoff(results=results, output_path=output_path, title=args.title, dpi=args.dpi)

    print(f"[INFO] Summary input: {summary_file}")
    print(f"[INFO] Plot output: {output_path}")


if __name__ == "__main__":
    main()