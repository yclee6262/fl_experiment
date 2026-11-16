"""Plot P3 candidate-exclusion and full-participant-removal pilot results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(paths, filename):
    rows = []
    for path in map(Path, paths):
        with (path / filename).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    effects = read(args.result_dirs, "participant_effects.csv")
    scenarios = read(args.result_dirs, "scenario_summary.csv")
    metrics = list(csv.DictReader(
        (Path(args.analysis_dir) / "scenario_metrics.csv").open()
    ))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, poison, title in (
        (axes[0, 0], "0.0", "Clean"),
        (axes[0, 1], "0.3", "Controlled poison"),
    ):
        selected = [row for row in effects if row["poison_ratio"] == poison]
        for row in selected:
            poisoned = row["is_poisoned"] == "True"
            ax.scatter(
                float(row["operational_internal_contribution"]),
                float(row["full_removal_external_contribution"]),
                s=65 if poisoned else 36,
                marker="X" if poisoned else "o",
                color="#E45756" if poisoned else "#4C78A8",
                alpha=0.8,
            )
        ax.axhline(0, color="#555555", linewidth=1)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_xlabel("operational candidate-only internal C_i")
        ax.set_ylabel("full-removal pooled-error contribution")
        ax.set_title(f"{title}: sign agreement by quadrant")
        ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for poison, offset, color, label in (
        ("0.0", -0.12, "#4C78A8", "Clean"),
        ("0.3", 0.12, "#E45756", "Poison"),
    ):
        selected = sorted(
            [row for row in metrics if row["poison_ratio"] == poison],
            key=lambda row: int(row["seed"]),
        )
        x = np.arange(len(selected)) + offset
        ax.bar(x, [float(row["operational_vs_full_spearman"]) for row in selected],
               width=0.24, color=color, label=label)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(np.arange(5), [str(i) for i in range(5)])
    ax.set_xlabel("seed")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Operational C_i ranking versus full-removal effect")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    groups = []
    candidate_cost = []
    full_cost = []
    for poison, label in (("0.0", "Clean"), ("0.3", "Poison")):
        selected = [row for row in scenarios if row["poison_ratio"] == poison]
        groups.append(label)
        candidate_cost.append(np.mean([int(row["candidate_exclusion_requests"])
                                       for row in selected]))
        full_cost.append(np.mean([int(row["full_removal_requests"])
                                  for row in selected]))
    x = np.arange(len(groups))
    width = 0.34
    ax.bar(x - width / 2, candidate_cost, width, label="candidate-only exclusion",
           color="#F58518")
    ax.bar(x + width / 2, full_cost, width, label="full participant removal",
           color="#54A24B")
    ax.set_xticks(x, groups)
    ax.set_ylabel("mean model-query requests per scenario")
    ax.set_title("Contribution-estimation cost")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("P3 five-seed contribution-validity pilot", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
