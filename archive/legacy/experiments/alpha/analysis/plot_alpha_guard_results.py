"""Create report-ready figures for the E_alpha_9 guard-validation experiments."""

from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
        if (candidate / "project_support/legacy.py").is_file():
            from project_support.legacy import prepare
            return prepare()
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = _source_workspace(__file__)
DATA_ROOT = ROOT / "alpha_weight_outputs"
OUTPUT_DIR = ROOT / "report_figures" / "alpha_guard_validation"

COLORS = {
    "Baseline": "#7A7A7A",
    "Fixed": "#D95F59",
    "Guarded": "#247BA0",
}
MARKERS = {"Baseline": "o", "Fixed": "s", "Guarded": "^"}
METHODS = ["Baseline", "Fixed", "Guarded"]


def read_csv(directory, filename):
    path = DATA_ROOT / directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment file: {path}")
    return pd.read_csv(path)


def select_summary(frame, update_rate, update_policy, method):
    selected = frame[
        np.isclose(frame["update_rate"], update_rate)
        & (frame["update_policy"] == update_policy)
    ].copy()
    selected["method"] = method
    return selected


def load_scenario(clean):
    if clean:
        fixed = read_csv("e_alpha_9b_clean_fixed", "summary.csv")
        guarded = read_csv("e_alpha_9b_clean_guarded", "summary.csv")
    else:
        fixed = read_csv("e_alpha_9c_poison_fixed", "summary.csv")
        guarded = read_csv("e_alpha_9c_poison_guarded", "summary.csv")

    parts = [
        select_summary(fixed, 0.0, "fixed", "Baseline"),
        select_summary(fixed, 0.5, "fixed", "Fixed"),
        select_summary(guarded, 0.5, "guarded", "Guarded"),
    ]
    combined = pd.concat(parts, ignore_index=True)
    numeric = [
        "seed",
        "final_evaluation_loss",
        "final_target_error",
        "final_consistency_l1",
        "rounds_executed",
        "updates_accepted",
        "gate_triggers",
    ]
    for column in numeric:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def plot_paired_leval(clean, poison):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    for ax, frame, title in zip(
        axes,
        [clean, poison],
        ["Clean coalition (n = 30)", "Controlled poisoned coalition (n = 20)"],
    ):
        pivot = frame.pivot(index="seed", columns="method", values="final_evaluation_loss")
        pivot = pivot[METHODS].dropna()
        x = np.arange(len(METHODS))
        for _, row in pivot.iterrows():
            ax.plot(x, row.values, color="#B8B8B8", alpha=0.42, linewidth=0.8, zorder=1)
        for index, method in enumerate(METHODS):
            values = pivot[method].to_numpy()
            ax.scatter(
                np.full(len(values), index),
                values,
                s=25,
                color=COLORS[method],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
            mean = float(np.mean(values))
            ax.scatter(
                index,
                mean,
                marker="D",
                s=72,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                f"mean {mean:.4f}",
                (index, mean),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )
        ax.set_xticks(x, METHODS)
        ax.set_ylabel("Final evaluation loss (L_eval)")
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
        style_axis(ax)

    fig.suptitle(
        "Paired final evaluation loss across alpha-update policies",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.015,
        "Each gray line connects the same random seed. Diamonds show group means; lower is better.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, "figure1_paired_final_evaluation_loss")


def plot_tradeoff(clean, poison):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    for ax, frame, title in zip(
        axes,
        [clean, poison],
        ["Clean coalition", "Controlled poisoned coalition"],
    ):
        for method in METHODS:
            group = frame[frame["method"] == method].dropna(
                subset=["final_consistency_l1", "final_evaluation_loss"]
            )
            ax.scatter(
                group["final_consistency_l1"],
                group["final_evaluation_loss"],
                s=38,
                marker=MARKERS[method],
                color=COLORS[method],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.4,
                label=f"{method} (n = {len(group)})",
            )
            if len(group):
                ax.scatter(
                    group["final_consistency_l1"].median(),
                    group["final_evaluation_loss"].median(),
                    s=115,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=4,
                )
        ax.set_xlabel("Final alpha-q L1 distance")
        ax.set_ylabel("Final evaluation loss (L_eval)")
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.text(
            0.02,
            0.97,
            "Preferred region: lower left",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
        style_axis(ax)

    fig.suptitle(
        "Trade-off between contribution consistency and solution quality",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.015,
        "Only runs with an informative final contribution distribution are shown; enlarged markers are medians.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, "figure2_consistency_quality_tradeoff")


def plot_seed12_trajectory():
    fixed_rounds = read_csv("e_alpha_9c_poison_fixed", "rounds.csv")
    guarded_rounds = read_csv("e_alpha_9c_poison_guarded", "rounds.csv")
    fixed_summary = read_csv("e_alpha_9c_poison_fixed", "summary.csv")
    guarded_summary = read_csv("e_alpha_9c_poison_guarded", "summary.csv")

    fixed = fixed_rounds[
        (fixed_rounds["seed"] == 12)
        & np.isclose(fixed_rounds["update_rate"], 0.5)
        & (fixed_rounds["update_policy"] == "fixed")
    ].copy()
    guarded = guarded_rounds[
        (guarded_rounds["seed"] == 12)
        & np.isclose(guarded_rounds["update_rate"], 0.5)
        & (guarded_rounds["update_policy"] == "guarded")
    ].copy()
    baseline_row = fixed_summary[
        (fixed_summary["seed"] == 12)
        & np.isclose(fixed_summary["update_rate"], 0.0)
        & (fixed_summary["update_policy"] == "fixed")
    ].iloc[0]
    guarded_final = guarded_summary[
        (guarded_summary["seed"] == 12)
        & np.isclose(guarded_summary["update_rate"], 0.5)
        & (guarded_summary["update_policy"] == "guarded")
    ].iloc[0]

    for frame in [fixed, guarded]:
        for column in ["round", "evaluation_loss", "consistency_l1"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["current_state_accepted", "gate_triggered"]:
        if guarded[column].dtype != bool:
            guarded[column] = guarded[column].astype(str).str.lower().eq("true")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    metrics = [
        ("evaluation_loss", "Evaluation loss (L_eval)"),
        ("consistency_l1", "Alpha-q L1 distance"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(
            fixed["round"],
            fixed[metric],
            color=COLORS["Fixed"],
            linewidth=2.0,
            marker="s",
            markersize=4,
            label="Fixed update",
        )
        ax.plot(
            guarded["round"],
            guarded[metric],
            color=COLORS["Guarded"],
            linewidth=1.5,
            alpha=0.75,
            label="Guarded candidates",
        )
        accepted = guarded[guarded["current_state_accepted"]]
        rejected = guarded[guarded["gate_triggered"]]
        ax.scatter(
            accepted["round"],
            accepted[metric],
            s=38,
            marker="o",
            color=COLORS["Guarded"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="Accepted state",
        )
        ax.scatter(
            rejected["round"],
            rejected[metric],
            s=48,
            marker="x",
            color="#B2182B",
            linewidth=1.4,
            zorder=5,
            label="Rejected by gate",
        )
        final_round = int(guarded_final["final_accepted_round"])
        final_point = guarded[guarded["round"] == final_round]
        ax.scatter(
            final_point["round"],
            final_point[metric],
            s=155,
            marker="*",
            color="#F2C14E",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
            label="Final accepted state",
        )
        if metric == "evaluation_loss":
            ax.axhline(
                float(baseline_row["final_evaluation_loss"]),
                color=COLORS["Baseline"],
                linestyle="--",
                linewidth=1.4,
                label="No-update baseline",
            )
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
        style_axis(ax)

    axes[1].set_xlabel("Calibration round")
    axes[1].set_xticks(np.arange(0, 20, 2))
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Guarded calibration trajectory: poisoned coalition, seed 12",
        fontsize=13,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        "Guarded: 16 gate rejections, 3 accepted updates, final accepted round = 6.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, hspace=0.06)
    save_figure(fig, "figure3_seed12_guard_trajectory")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    clean = load_scenario(clean=True)
    poison = load_scenario(clean=False)
    plot_paired_leval(clean, poison)
    plot_tradeoff(clean, poison)
    plot_seed12_trajectory()
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
