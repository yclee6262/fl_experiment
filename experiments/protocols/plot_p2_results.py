"""Create the required P2 formal-comparison figures from saved CSV results."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "A_uniform_no_update",
    "B_stage0_no_update",
    "C_uniform_fixed",
    "D_uniform_guarded",
)
COLORS = {
    "A_uniform_no_update": "#4C78A8",
    "B_stage0_no_update": "#F58518",
    "C_uniform_fixed": "#E45756",
    "D_uniform_guarded": "#54A24B",
}
SHORT = dict(zip(METHODS, ("A", "B", "C", "D")))


def load_rows(result_dirs):
    rows = []
    for directory in map(Path, result_dirs):
        with (directory / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row, field):
    value = row.get(field, "")
    if value == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def paired_deltas(rows, metric):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["seed"], row["poison_ratio"])
        grouped[key][row["method_label"]] = row
    output = defaultdict(list)
    for (seed, poison), methods in grouped.items():
        baseline = number(methods[METHODS[0]], metric)
        for method in METHODS[1:]:
            value = number(methods[method], metric)
            if baseline is not None and value is not None:
                output[(poison, method)].append(value - baseline)
    return output


def plot_summary(rows, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    rng = np.random.default_rng(20260922)

    for ax, metric, title in (
        (axes[0, 0], "external_global_error", "Primary: paired pooled-evaluator error delta"),
        (axes[0, 1], "final_target_error", "Diagnostic: paired true-function error delta"),
    ):
        deltas = paired_deltas(rows, metric)
        positions, labels = [], []
        position = 0
        for poison in ("0.0", "0.3"):
            for method in METHODS[1:]:
                values = np.asarray(deltas[(poison, method)])
                positions.append(position)
                labels.append(f"{'Clean' if poison == '0.0' else 'Poison'}\n{SHORT[method]}")
                jitter = rng.uniform(-0.12, 0.12, len(values))
                ax.scatter(position + jitter, values, s=18, alpha=0.55,
                           color=COLORS[method])
                ax.boxplot(values, positions=[position], widths=0.5, showfliers=False,
                           medianprops={"color": "black"},
                           boxprops={"color": COLORS[method]},
                           whiskerprops={"color": COLORS[method]},
                           capprops={"color": COLORS[method]})
                position += 1
            position += 0.7
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("method − A error (lower is better)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    for poison, marker, scenario in (("0.0", "o", "Clean"), ("0.3", "s", "Poison")):
        selected = [row for row in rows if row["poison_ratio"] == poison]
        for method in METHODS:
            method_rows = [row for row in selected if row["method_label"] == method]
            requests = [sum(int(row[name]) for name in
                            ("stage3_requests", "contribution_requests", "payment_requests"))
                        for row in method_rows]
            errors = [float(row["external_global_error"]) for row in method_rows]
            ax.scatter(np.mean(requests), np.mean(errors), marker=marker, s=95,
                       color=COLORS[method], edgecolor="white", linewidth=0.8)
            ax.annotate(f"{scenario} {SHORT[method]}",
                        (np.mean(requests), np.mean(errors)), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mean total requests (log scale)")
    ax.set_ylabel("mean pooled-evaluator error")
    ax.set_title("Quality versus query cost")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for poison, marker in (("0.0", "o"), ("0.3", "s")):
        for method in METHODS:
            selected = [row for row in rows
                        if row["poison_ratio"] == poison
                        and row["method_label"] == method]
            x, y = [], []
            for row in selected:
                consistency = number(row, "final_consistency_l1")
                if consistency is not None:
                    x.append(consistency)
                    y.append(float(row["external_global_error"]))
            ax.scatter(x, y, marker=marker, s=24, alpha=0.5,
                       color=COLORS[method], label=f"{poison} {SHORT[method]}")
    ax.set_xlabel("final alpha-q L1")
    ax.set_ylabel("pooled-evaluator error")
    ax.set_title("Alpha-q agreement versus delivered-X quality")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=7, title="poison ratio / method")

    fig.suptitle("P2 formal comparison: seeds 100–129", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_outcomes(rows, output_path):
    groups = [(p, m) for p in ("0.0", "0.3") for m in METHODS]
    reasons = sorted({row["stop_reason"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    cmap = plt.get_cmap("tab20")
    for index, reason in enumerate(reasons):
        values = []
        for poison, method in groups:
            selected = [row for row in rows if row["poison_ratio"] == poison
                        and row["method_label"] == method]
            values.append(sum(row["stop_reason"] == reason for row in selected))
        axes[0].bar(x, values, bottom=bottom, label=reason, color=cmap(index))
        bottom += np.asarray(values)
    labels = [f"{'Clean' if p == '0.0' else 'Poison'}\n{SHORT[m]}" for p, m in groups]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("seeds")
    axes[0].set_title("Stopping reasons (all 30 seeds retained)")
    axes[0].legend(fontsize=7)

    infeasible, invalid_q = [], []
    for poison, method in groups:
        selected = [row for row in rows if row["poison_ratio"] == poison
                    and row["method_label"] == method]
        infeasible.append(sum(row["final_box_feasible"] != "True" for row in selected))
        invalid_q.append(sum(row.get("final_contribution_weights", "[]") in ("", "[]")
                             for row in selected))
    width = 0.36
    axes[1].bar(x - width / 2, infeasible, width, label="box infeasible", color="#E45756")
    axes[1].bar(x + width / 2, invalid_q, width, label="invalid q", color="#B279A2")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("seeds")
    axes[1].set_title("Failure and undefined-contribution counts")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("formal_summary.png", "formal_outcomes.png"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")
    rows = load_rows(args.result_dirs)
    plot_summary(rows, output_dir / "formal_summary.png")
    plot_outcomes(rows, output_dir / "formal_outcomes.png")
    print(f"Plotted {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
