"""18조합(3압축 x 2FL x 3α) 결과 시각화 템플릿.

results/logs/*_rounds.jsonl에서 실제 실행 결과를 읽어 PPL/ROUGE-L/통신량/
수렴속도 도표를 results/figures/에 생성한다. 아직 실행된 로그가 하나도
없으면 형태를 미리 볼 수 있도록 샘플 데이터로 대체하고, 각 그림에
"SAMPLE DATA" 워터마크를 남긴다 — 실제 로그가 생기면 그대로 다시 실행하면
자동으로 진짜 데이터로 그려진다.

사용법:
    python scripts/plot_results.py

A visualization template for the 18-combination (3 compression x 2 FL x 3
alpha) results.

Reads actual run results from results/logs/*_rounds.jsonl and produces
PPL/ROUGE-L/communication/convergence-speed figures into results/figures/.
If no logs exist yet, falls back to sample data so the shape can be
previewed now, watermarking each figure "SAMPLE DATA" — once real logs
exist, re-running this script draws the real data automatically.

Usage:
    python scripts/plot_results.py
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_interaction import COMPRESSIONS, FLS, ALPHAS, build_run_name

FIG_DIR = "results/figures"

# 검증된 categorical 팔레트 (dataviz 스킬, 3-slot all-pairs 통과: blue/orange/aqua)
# Validated categorical palette (dataviz skill, 3-slot all-pairs pass: blue/orange/aqua)
COLORS = {
    "lora": "#2a78d6",
    "qlora_8bit": "#eb6834",
    "qlora_4bit": "#1baf7a",
}
LABELS = {
    "lora": "LoRA (uncompressed)",
    "qlora_8bit": "QLoRA 8-bit",
    "qlora_4bit": "QLoRA 4-bit",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_real_results() -> Optional[List[dict]]:
    """results/logs/에서 18개 실행 로그를 전부 읽는다. 하나라도 없으면 None.

    Reads all 18 run logs from results/logs/. Returns None if any are missing.
    """
    results = []
    for compression in COMPRESSIONS:
        for fl in FLS:
            for alpha in ALPHAS:
                run_name = build_run_name(compression, fl, alpha)
                path = f"results/logs/{run_name}_rounds.jsonl"
                if not os.path.exists(path):
                    return None
                with open(path) as f:
                    lines = [json.loads(line) for line in f]
                last = lines[-1]
                results.append({
                    "compression": compression, "fl": fl, "alpha": alpha,
                    "val_perplexity": last["val_perplexity"],
                    "rouge_l": last.get("rouge_l"),
                    "total_communication_bytes": sum(r["communication_bytes_this_round"] for r in lines),
                    "rounds_run": len(lines),
                })
    return results


def make_sample_results() -> List[dict]:
    """로그가 아직 없을 때 도표 형태를 미리 보기 위한 샘플 데이터.
    non-IID가 강할수록(alpha 작을수록) 압축 페널티가 커지는 가상 패턴.

    Sample data to preview the chart shapes before any real logs exist.
    A synthetic pattern where the compression penalty grows as non-IID
    intensifies (alpha shrinks)."""
    rng = random.Random(7)
    base_ppl = {0.1: 12.0, 1: 9.0, 10: 7.5}
    penalty_scale = {"lora": 0.0, "qlora_8bit": 0.4, "qlora_4bit": 1.0}
    results = []
    for compression in COMPRESSIONS:
        for fl in FLS:
            fl_offset = 0.0 if fl == "fedprox" else 0.6
            for alpha in ALPHAS:
                non_iid_amplifier = {0.1: 3.0, 1: 1.3, 10: 0.6}[alpha]
                ppl = base_ppl[alpha] + fl_offset + penalty_scale[compression] * non_iid_amplifier
                ppl += rng.uniform(-0.15, 0.15)
                rounds = max(3, round(6 + non_iid_amplifier + penalty_scale[compression] * 1.5 + {"lora": 0, "qlora_8bit": 0.4, "qlora_4bit": 0.8}[compression] + rng.uniform(-0.4, 0.4)))
                rouge = max(0.05, 0.42 - 0.10 * (penalty_scale[compression] * non_iid_amplifier / 3.0) + rng.uniform(-0.01, 0.01))
                payload_per_round = {"lora": 8_000_000, "qlora_8bit": 8_000_000, "qlora_4bit": 8_000_000}[compression]
                results.append({
                    "compression": compression, "fl": fl, "alpha": alpha,
                    "val_perplexity": round(ppl, 4),
                    "rouge_l": round(rouge, 4),
                    "total_communication_bytes": payload_per_round * rounds * 8,
                    "rounds_run": rounds,
                })
    return results


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def _watermark(fig, is_sample: bool):
    if is_sample:
        fig.text(0.99, 0.01, "SAMPLE DATA — placeholder until results/logs/ is populated",
                  ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")


def plot_trend(results: List[dict], field: str, ylabel: str, title: str, filename: str, is_sample: bool, higher_is_better: bool = False):
    """alpha(x축) x compression(선) x FL(패널) 트렌드 라인 차트.

    A trend line chart: alpha (x-axis) x compression (line) x FL (panel)."""
    fig, axes = plt.subplots(1, len(FLS), figsize=(10, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, fl in zip(axes, FLS):
        _style_axes(ax)
        for compression in COMPRESSIONS:
            ys = [next(r[field] for r in results if r["compression"] == compression and r["fl"] == fl and r["alpha"] == a) for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker="o", markersize=6, linewidth=2, color=COLORS[compression], label=LABELS[compression])
            # alpha=0.1(격자 오른쪽, invert_xaxis 이후) 지점에 라벨 — 선이 가장 벌어지는 지점
            # Label at alpha=0.1 (right side after invert_xaxis) — where lines are most spread out
            ax.annotate(LABELS[compression], xy=(ALPHAS[0], ys[0]), xytext=(-6, 0),
                        textcoords="offset points", fontsize=8.5, color=COLORS[compression], va="center", ha="right")
        ax.set_title(fl.upper(), fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Dirichlet α (non-IID intensity →)", fontsize=9.5, color=MUTED)
        ax.set_xticks(ALPHAS)
        ax.invert_xaxis()  # 왼쪽(near-IID, alpha=10) -> 오른쪽(강한 non-IID, alpha=0.1)
        ax.margins(x=0.18)
    axes[0].set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    fig.suptitle(title, fontsize=13, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _watermark(fig, is_sample)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_grouped_bar(results: List[dict], field: str, ylabel: str, title: str, filename: str, is_sample: bool):
    """alpha(x축 그룹) x compression(막대) x FL(패널) 그룹 막대 차트.

    A grouped bar chart: alpha (x-axis groups) x compression (bars) x FL (panel)."""
    fig, axes = plt.subplots(1, len(FLS), figsize=(10, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    width = 0.25
    x = range(len(ALPHAS))
    for ax, fl in zip(axes, FLS):
        _style_axes(ax)
        for i, compression in enumerate(COMPRESSIONS):
            ys = [next(r[field] for r in results if r["compression"] == compression and r["fl"] == fl and r["alpha"] == a) for a in ALPHAS]
            offsets = [xi + (i - 1) * width for xi in x]
            bars = ax.bar(offsets, ys, width=width * 0.92, color=COLORS[compression], label=LABELS[compression], zorder=3)
            ax.bar_label(bars, fmt="%.2f" if max(ys) < 100 else "%.2g", fontsize=7, color=INK, padding=2)
        ax.set_title(fl.upper(), fontsize=11, color=INK, loc="left")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=9.5, color=MUTED)
    axes[0].set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    fig.suptitle(title, fontsize=13, color=INK, x=0.02, y=0.99, ha="left", va="top", fontweight="bold")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.5, 0.90))
    fig.tight_layout(rect=[0, 0.03, 1, 0.82])
    _watermark(fig, is_sample)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary_table(results: List[dict], is_sample: bool):
    path = os.path.join(FIG_DIR, "summary_table.md")
    with open(path, "w") as f:
        f.write("# 18조합 결과 요약" + (" (SAMPLE DATA)" if is_sample else "") + "\n\n")
        f.write("| compression | fl | alpha | val_perplexity | rouge_l | rounds_run | total_communication_bytes |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for compression in COMPRESSIONS:
            for fl in FLS:
                for alpha in ALPHAS:
                    r = next(x for x in results if x["compression"] == compression and x["fl"] == fl and x["alpha"] == alpha)
                    f.write(f"| {compression} | {fl} | {alpha} | {r['val_perplexity']:.4f} | "
                            f"{r['rouge_l']:.4f} | {r['rounds_run']} | {r['total_communication_bytes']:,} |\n")
    print(f"summary table -> {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    results = load_real_results()
    is_sample = results is None
    if is_sample:
        print("results/logs/에 18개 로그가 아직 없어 샘플 데이터로 미리보기를 생성합니다.")
        print("(No real logs found in results/logs/ yet — generating a preview with sample data.)")
        results = make_sample_results()

    plot_trend(results, "val_perplexity", "Validation Perplexity (↓ better)",
               "PPL vs. Dirichlet α, by compression level", "ppl_vs_alpha.png", is_sample)
    plot_trend(results, "rounds_run", "Rounds to Converge (↓ better, free metric)",
               "Convergence speed vs. Dirichlet α, by compression level", "convergence_vs_alpha.png", is_sample)
    plot_grouped_bar(results, "rouge_l", "ROUGE-L (↑ better)",
                      "ROUGE-L by compression level and α (final round only)", "rouge_l_bars.png", is_sample)
    plot_grouped_bar(results, "total_communication_bytes", "Total Communication (bytes, ↓ better)",
                      "Communication cost by compression level and α", "communication_bars.png", is_sample)
    write_summary_table(results, is_sample)

    print(f"\n4 figures + 1 summary table written to {FIG_DIR}/")
    if is_sample:
        print("실제 GPU 실행 결과가 results/logs/에 쌓이면 이 스크립트를 다시 돌리세요 — 자동으로 진짜 데이터로 교체됩니다.")
        print("(Once real GPU results land in results/logs/, re-run this script — it will automatically switch to the real data.)")


if __name__ == "__main__":
    main()
