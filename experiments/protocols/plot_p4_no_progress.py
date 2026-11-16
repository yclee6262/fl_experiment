"""Plot paired P4 no-progress control and treatment costs."""

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
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with (Path(args.analysis_dir) / "paired_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    formal = len(rows) >= 60
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, poison, title in (
        (axes[0], "0.0", "Clean"),
        (axes[1], "0.3", "Controlled poison"),
    ):
        selected = [row for row in rows if row["poison_ratio"] == poison]
        x = np.arange(len(selected))
        width = 0.36
        control = [int(row["control_requests"]) for row in selected]
        treatment = [int(row["treatment_requests"]) for row in selected]
        ax.bar(x - width / 2, control, width, label="control", color="#9D9D9D")
        ax.bar(x + width / 2, treatment, width, label="no-progress stop", color="#54A24B")
        for index, row in enumerate(selected):
            if not formal and row["no_progress_triggered"] == "True":
                ax.text(index, max(control[index], treatment[index]) * 1.02, "triggered",
                        ha="center", va="bottom", fontsize=8, color="#2E6E2E")
        ax.set_xticks(x, [row["seed"] for row in selected], rotation=90 if formal else 0)
        ax.set_xlabel("seed")
        ax.set_ylabel("total model-query requests")
        ax.set_title(f"{title}: accepted X identical for every pair")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle(
        "P4 guarded no-progress stopping formal comparison"
        if formal else "P4 guarded no-progress stopping pilot",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
