"""Summarize the paired ten-rule cohort from saved outputs; no model execution."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[3]
LABELS = dict(A_uniform="U", B_stage0="Q", stacked_full="E", ensemble_shrink="EU",
              B_stacked_half="EQ", pure_support="S", peer_support_shrink="SU",
              support_quality_mix50="SQ", support_ensemble_mix50="SE", quality_shrink="QU")
SCENARIOS = ["reference_clean", "low_quality_25pct", "sample_imbalance_4x"]

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write_csv(path, data):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)

def summarize(output, figure=None):
    source = ROOT / "alpha_weight_outputs/se_qu_extension_v1_20261107"
    reg, analysis = read(source / "registration.json"), read(source / "analysis.json")
    assert read(source / "acceptance.json")["passed"]
    assert read(source / "supplementary_audit.json")["passed"]
    hashes = {}
    for name, expected in reg["original_results_sha256"].items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert digest == expected, name
        hashes[name] = digest
    index = {}
    seeds = []
    for batch in reg["batches"][:3]:
        seeds.extend(batch["seeds"])
        for kind in ["six_method_results", "pure_results", "results"]:
            path = ROOT / batch[kind] / "results.jsonl"
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert read(path.parent / "verification.json")["passed"]
            for row in rows(path):
                key = (row["scenario"], row["seed"], row["method"])
                assert key not in index
                assert row["status"] == "valid"
                assert all(math.isfinite(v) and abs(v) <= 1 + 1e-8 for v in row["X"])
                assert all(math.isfinite(v) and v >= -1e-10 for v in row["alpha"])
                assert abs(sum(row["alpha"]) - 1) < 1e-8
                assert row["agent_ids"] == [1, 2, 20]
                assert row["additional_labels"] == 0
                index[key] = row
    assert len(index) == 2700
    assert set(index) == set(itertools.product(SCENARIOS, seeds, LABELS))
    component_max = 0
    for scenario, seed in itertools.product(SCENARIOS, seeds):
        for method, left, right in [("support_ensemble_mix50", "pure_support", "stacked_full"),
                                     ("quality_shrink", "B_stage0", "A_uniform")]:
            get = lambda m: index[scenario, seed, m]["alpha"]
            error = max(abs(v - (a + b)/2) for v, a, b in zip(get(method), get(left), get(right)))
            component_max = max(component_max, error)
    assert component_max < 1e-12
    # Verify the stored decisions were sealed without pooled scores, and replay agrees.
    for batch in reg["batches"][:3]:
        for row in rows(ROOT / batch["results"] / "decisions.jsonl"):
            assert row.get("pooled_error") is None
            assert row["X"] == index[row["scenario"], row["seed"], row["method"]]["X"]
    replay_count = 0
    for batch in reg["batches"][3:]:
        assert read(ROOT / batch["results"] / "verification.json")["passed"]
        for row in rows(ROOT / batch["results"] / "results.jsonl"):
            original = index[row["scenario"], row["seed"], row["method"]]
            for field in ["alpha", "X", "pooled_error", "truth_error"]:
                assert row[field] == original[field]
            replay_count += 1
    assert replay_count == 18
    lookup = {(g["method"], g["endpoint"]): g["mean"] for g in analysis["groups"]}
    summary = []
    def fallback(audit):
        if not isinstance(audit, dict):
            return False
        return bool(audit.get("fallback_reason")) or any(fallback(v) for v in audit.values())
    for method, label in LABELS.items():
        selected = [r for r in index.values() if r["method"] == method]
        by_seed = [statistics.mean(index[sc, seed, method]["pooled_error"] for sc in SCENARIOS)
                   for seed in seeds]
        entry = {"method": label, "overall": statistics.mean(by_seed)}
        for sc in SCENARIOS:
            entry[sc] = statistics.mean(index[sc, seed, method]["pooled_error"] for seed in seeds)
        for endpoint, value in [("equal_scenario_mean", entry["overall"]),
                                *[(sc, entry[sc]) for sc in SCENARIOS]]:
            assert abs(value - lookup[method, endpoint]) < 1e-15
        entry.update(valid=len(selected), failed=0, fallbacks=sum(fallback(r["alpha_audit"]) for r in selected),
                     mean_stage3_requests=statistics.mean(r["stage3_requests"] for r in selected),
                     required_peer_queries=6 if label in {"S", "SU", "SQ", "SE"} else 0)
        assert entry["fallbacks"] == 0
        assert abs(entry["mean_stage3_requests"] - analysis["costs"][method]["mean_stage3_requests"]) < 1e-12
        summary.append(entry)
    summary.sort(key=lambda r: r["overall"])
    comparisons = []
    for c in analysis["comparisons"]:
        delta = statistics.mean(statistics.mean(
            index[sc, seed, c["method"]]["pooled_error"] - index[sc, seed, c["baseline"]]["pooled_error"]
            for sc in SCENARIOS) for seed in seeds)
        assert abs(delta - c["mean_delta"]) < 1e-15
        assert c["CI17"][0] < 0 < c["CI17"][1]
        assert not c["stable_in_this_cohort"]
        comparisons.append(dict(contrast=LABELS[c["method"]]+" - "+LABELS[c["baseline"]],
            mean_delta=delta, ci17_low=c["CI17"][0], ci17_high=c["CI17"][1],
            stable=c["stable_in_this_cohort"]))
    assert len(comparisons) == 17
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "means_and_queries.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    audit = dict(rows=len(index), seeds=len(seeds), scenarios=SCENARIOS,
        methods=LABELS, replay_rows=replay_count, max_component_difference=component_max,
        source_sha256=hashes, statistics="Paired historical cohort; 17-contrast bootstrap intervals reused from frozen analysis; no new bootstrap or experiment.")
    (output / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    table = ["# 十種固定 alpha 初始化：資料彙整", "",
        "四個基礎 U、Q、E、S 與全部六個等比例兩兩混合：EU、EQ、SU、SQ、SE、QU。混合的是 alpha 向量，然後執行一次 Stage 3 best-of 搜尋。", "",
        "沿用 90 seeds × 三情境，完整 2700 筆、每法 270 筆有效、無 fallback。新增 SE/QU 為既有案例的配對補充比較；原六法的 12-contrast family 與 SE/QU 的 17-contrast family 分開解讀。", "",
        "|方法|整體平均|乾淨|低品質|不平衡|Stage 3 平均查詢數|", "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        table.append("|"+r["method"]+"|"+"|".join(f"{r[k]:.8f}" for k in ["overall", *SCENARIOS])+f"|{r['mean_stage3_requests']:.2f}|")
    table += ["", "EU 整體平均最低；SE、QU、E 分別在乾淨、低品質、不平衡情境平均最低。SE/QU 相關 17 個校正區間均跨零，沒有比較通過其穩定改善條件。", "",
        "查詢數取自記錄，可直接彙整；執行時間分批測量，不作同時負載下的速度排名。SE 額外需要六個非自身 proposal 評估，QU 不需要支持查詢。", "",
        "來源：alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json、analysis.json、acceptance.json、supplementary_audit.json 及註冊中指定的原始結果。", "",
        "彙整程式重新驗證完整案例覆蓋、原八法檔案雜湊、數值均值、alpha 組成、保存決策與 18 筆重播；沿用凍結統計區間，沒有重訓或新增求解。", "",
        "means_and_queries.csv 包含十法各情境均值與查詢成本；paired_comparisons.csv 包含 17 個補充配對區間；audit.json 保存來源雜湊。"]
    (output / "README.md").write_text("\n".join(table)+"\n")
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42})
        fig, ax = plt.subplots(figsize=(7.0, 3.9))
        x = np.arange(len(summary)); width = .25
        for j, (sc, label, color) in enumerate(zip(SCENARIOS,
                ["Clean", "Low quality", "Sample imbalance"], ["#3376A5", "#D78A38", "#46988E"])):
            ax.bar(x+(j-1)*width, [r[sc] for r in summary], width, label=label, color=color)
        ax.set_xticks(x, [r["method"] for r in summary]); ax.set_xlabel("Initialization rule")
        ax.set_ylabel("Mean pooled absolute target error")
        ax.set_ylim(0, .026); ax.legend(loc="upper center", ncol=3, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=.2)
        fig.tight_layout(); figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure); plt.close(fig)
    print(json.dumps(dict(rows=len(index), means={r["method"]:r["overall"] for r in summary}, replay_rows=replay_count)))
    return summary, comparisons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/alpha_initialization_ten_methods")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    summarize(args.output, args.figure)
