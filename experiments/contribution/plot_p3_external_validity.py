"""Create the formal P3 external-validity summary figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--controlled-poison", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = {
        "Clean": read(Path(args.clean) / "participant_effects.csv"),
        "Controlled poison": read(Path(args.controlled_poison) / "participant_effects.csv"),
    }
    seed_metrics = read(Path(args.analysis) / "seed_metrics.csv")
    aggregate = {row["scenario"]: row for row in read(Path(args.analysis) / "aggregate_metrics.csv")}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {"Clean": "#2878B5", "Controlled poison": "#D95F02"}
    for axis, (label, rows) in zip(axes[0], sources.items()):
        x = np.asarray([float(row["operational_internal_contribution"]) for row in rows])
        y = np.asarray([float(row["full_removal_external_contribution"]) for row in rows])
        axis.scatter(x, y, s=23, alpha=0.7, color=colors[label], edgecolor="none")
        axis.axhline(0, color="0.55", lw=0.8)
        axis.axvline(0, color="0.55", lw=0.8)
        axis.set_title(f"{label}: participant effects")
        axis.set_xlabel("Operational internal contribution")
        axis.set_ylabel("Full-removal pooled contribution")

    scenario_labels = (("clean", "Clean"), ("controlled_poison", "Controlled poison"))
    rho_data = [
        [float(row["spearman_operational_vs_full"]) for row in seed_metrics
         if row["scenario"] == scenario and row["spearman_operational_vs_full"]]
        for scenario, _ in scenario_labels
    ]
    axes[1, 0].boxplot(rho_data, labels=[label for _, label in scenario_labels],
                       showmeans=True)
    axes[1, 0].axhline(0, color="0.55", lw=0.8)
    axes[1, 0].axhline(0.5, color="#2E8B57", lw=1, ls="--", label="practical threshold")
    axes[1, 0].set_ylabel("Seed-level Spearman")
    axes[1, 0].set_title("Ranking agreement by seed")
    axes[1, 0].legend(frameon=False)

    points = [float(aggregate[scenario]["mean_direction_agreement"])
              for scenario, _ in scenario_labels]
    low = [float(aggregate[scenario]["simultaneous_ci_low_direction_agreement"])
           for scenario, _ in scenario_labels]
    high = [float(aggregate[scenario]["simultaneous_ci_high_direction_agreement"])
            for scenario, _ in scenario_labels]
    x = np.arange(2)
    axes[1, 1].bar(x, points, color=[colors[label] for _, label in scenario_labels], width=0.6)
    axes[1, 1].errorbar(x, points, yerr=[np.asarray(points)-low, np.asarray(high)-points],
                        fmt="none", ecolor="black", capsize=5)
    axes[1, 1].axhline(0.7, color="#2E8B57", lw=1, ls="--", label="practical threshold")
    axes[1, 1].axhline(0.5, color="0.55", lw=0.8, ls=":", label="chance reference")
    axes[1, 1].set_xticks(x, [label for _, label in scenario_labels])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Direction agreement")
    axes[1, 1].set_title("Mean and 98.75% simultaneous CI")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("P3 formal external-validity test (seeds 3600–3629)", fontsize=15)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
