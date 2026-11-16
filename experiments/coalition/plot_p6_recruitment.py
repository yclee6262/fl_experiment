"""Plot P6 budget/eta recruitment pilot results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with (Path(args.analysis_dir) / "method_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    methods = ("exact", "objective_greedy", "legacy_greedy")
    colors = ("#4c78a8", "#f58518", "#54a24b")
    conditions = sorted({
        (float(row["budget"]), float(row["diversity_eta"])) for row in rows
    })
    labels = [f"B={budget:g}\neta={eta:g}" for budget, eta in conditions]
    x = np.arange(len(conditions))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fields = (
        ("mean_objective", "Mean selection objective", False),
        ("mean_external_error", "Mean pooled error", False),
        ("mean_inversion_seconds", "Mean inversion seconds", True),
    )
    lookup = {
        (float(row["budget"]), float(row["diversity_eta"]), row["method"]): row
        for row in rows
    }
    for method_index, (method, color) in enumerate(zip(methods, colors)):
        offset = (method_index - 1) * width
        for axis, (field, title, log_scale) in zip(axes, fields):
            axis.bar(
                x + offset,
                [float(lookup[(*condition, method)][field]) for condition in conditions],
                width,
                color=color,
                label=method,
            )
            axis.set_title(title)
            if log_scale:
                axis.set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("P6 budget and diversity pilot (5 paired seeds)")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
