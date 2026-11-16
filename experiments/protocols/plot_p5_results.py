"""Create a compact P5 pilot quality, cost, and evaluator-reliability figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read(Path(args.analysis_dir) / "method_summary.csv")
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    methods = ("A_uniform_no_update", "D_uniform_guarded")
    labels = [name.replace("controlled_", "ctrl_").replace("_", "\n")
              for name in scenarios]
    x = np.arange(len(scenarios))
    width = 0.38
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for offset, method in zip((-width / 2, width / 2), methods):
        selected = {row["scenario"]: row for row in rows if row["method"] == method}
        axes[0].bar(x + offset, [float(selected[s]["mean_external_error"])
                                for s in scenarios], width, label=method)
        axes[1].bar(x + offset, [float(selected[s]["mean_requests"])
                                for s in scenarios], width)
    evaluator = {row["scenario"]: row for row in rows
                 if row["method"] == methods[0]}
    axes[2].bar(x, [np.sqrt(float(evaluator[s]["mean_evaluator_truth_mse"]))
                    for s in scenarios], 0.55, color="#5b8e7d")
    axes[0].set_ylabel("Mean pooled error")
    axes[1].set_ylabel("Mean API requests")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Evaluator truth RMSE")
    axes[2].set_xticks(x, labels, fontsize=8)
    axes[0].legend(frameon=False)
    axes[0].set_title("P5 one-factor pilot (5 paired seeds; descriptive only)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
