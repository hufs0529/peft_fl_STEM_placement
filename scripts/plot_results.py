"""24조합(3압축 x 2FL x 4α) 결과 시각화 템플릿.

results/logs/*_rounds.jsonl에서 실제 실행 결과를 읽어 PPL/ROUGE-L/통신량/
수렴속도/**peak VRAM/총 학습 시간** 도표를 results/figures/에 생성한다.
아직 실행된 로그가 하나도 없으면 형태를 미리 볼 수 있도록 샘플 데이터로
대체하고, 각 그림에 "SAMPLE DATA" 워터마크를 남긴다 — 실제 로그가 생기면
그대로 다시 실행하면 자동으로 진짜 데이터로 그려진다.

peak_vram_gb는 alpha와 거의 무관(압축 강도에만 좌우)해야 정상이므로,
`peak_vram_vs_alpha.png`에서 선이 평평한지가 그 자체로 "압축의 메모리
이점이 non-IID 강도와 독립적인가"를 검증하는 시각적 체크가 된다.

사용법:
    python scripts/plot_results.py

A visualization template for the 24-combination (3 compression x 2 FL x 4
alpha) results.

Reads actual run results from results/logs/*_rounds.jsonl and produces
PPL/ROUGE-L/communication/convergence-speed/**peak VRAM/total training
latency** figures into results/figures/. If no logs exist yet, falls back
to sample data so the shape can be previewed now, watermarking each figure
"SAMPLE DATA" — once real logs exist, re-running this script draws the
real data automatically.

peak_vram_gb should be roughly independent of alpha (driven only by
compression level), so whether the lines in `peak_vram_vs_alpha.png` are
flat is itself a visual check of "is the compression memory benefit
independent of non-IID intensity".

Usage:
    python scripts/plot_results.py
"""

import json
import os
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_interaction import COMPRESSIONS, FLS, ALPHAS, N_COMBOS, build_run_name

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
    """results/logs/에서 24개 실행 로그를 전부 읽는다. 하나라도 없으면 None.

    Reads all 24 run logs from results/logs/. Returns None if any are missing.
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
                vram_values = [r["peak_vram_gb"] for r in lines if r.get("peak_vram_gb") is not None]
                results.append({
                    "compression": compression, "fl": fl, "alpha": alpha,
                    "val_perplexity": last["val_perplexity"],
                    "rouge_l": last.get("rouge_l"),
                    "total_communication_bytes": sum(r["communication_bytes_this_round"] for r in lines),
                    "rounds_run": len(lines),
                    "total_latency_sec": sum(r["round_latency_sec"] for r in lines),
                    "peak_vram_gb": sum(vram_values) / len(vram_values) if vram_values else None,
                })
    return results


def make_sample_results() -> List[dict]:
    """로그가 아직 없을 때 도표 형태를 미리 보기 위한 샘플 데이터.
    non-IID가 강할수록(alpha 작을수록) 압축 페널티가 커지는 가상 패턴.

    Sample data to preview the chart shapes before any real logs exist.
    A synthetic pattern where the compression penalty grows as non-IID
    intensifies (alpha shrinks)."""
    rng = random.Random(7)
    # ALPHAS에 값이 추가돼도 KeyError가 나지 않도록 dict 조회 대신 보간식을
    # 쓴다 — alpha가 커질수록(IID에 가까울수록) PPL과 증폭계수가 함께 낮아지는
    # 형태만 유지하면 되는 미리보기용 합성 데이터다.
    # Use a formula instead of a dict lookup so adding values to ALPHAS never
    # raises KeyError — this is preview-only synthetic data, and all it needs
    # to preserve is the shape: larger alpha (closer to IID) means lower PPL
    # and a smaller amplifier.
    def _base_ppl(a: float) -> float:
        return 7.0 + 2.5 / (1.0 + math.log10(max(a, 1e-6)) + 1.5)

    def _amplifier(a: float) -> float:
        return 0.4 + 2.6 / (1.0 + math.log10(max(a, 1e-6)) + 1.5)

    penalty_scale = {"lora": 0.0, "qlora_8bit": 0.4, "qlora_4bit": 1.0}
    # peak_vram_gb: 압축이 강할수록 낮고(양자화가 base 모델 메모리를 줄임),
    # alpha와는 거의 무관해야 정상 — 상관관계가 보이면 오히려 이상 신호.
    # peak_vram_gb: lower for heavier compression (quantization shrinks the
    # base model's memory footprint), and should be ~flat across alpha — a
    # visible alpha trend here would itself be a red flag.
    base_vram_gb = {"lora": 6.5, "qlora_8bit": 4.0, "qlora_4bit": 2.5}
    # 라운드당 연산 오버헤드: 양자화된 가중치를 매 스텝 dequant해야 해서
    # lora < qlora_8bit < qlora_4bit 순으로 느려짐.
    # Per-round compute overhead: dequantizing weights every step makes
    # this slower in the order lora < qlora_8bit < qlora_4bit.
    per_round_latency_sec = {"lora": 45.0, "qlora_8bit": 58.0, "qlora_4bit": 70.0}
    results = []
    for compression in COMPRESSIONS:
        for fl in FLS:
            fl_offset = 0.0 if fl == "fedprox" else 0.6
            for alpha in ALPHAS:
                non_iid_amplifier = _amplifier(alpha)
                ppl = _base_ppl(alpha) + fl_offset + penalty_scale[compression] * non_iid_amplifier
                ppl += rng.uniform(-0.15, 0.15)
                rounds = max(3, round(6 + non_iid_amplifier + penalty_scale[compression] * 1.5 + {"lora": 0, "qlora_8bit": 0.4, "qlora_4bit": 0.8}[compression] + rng.uniform(-0.4, 0.4)))
                rouge = max(0.05, 0.42 - 0.10 * (penalty_scale[compression] * non_iid_amplifier / 3.0) + rng.uniform(-0.01, 0.01))
                payload_per_round = {"lora": 8_000_000, "qlora_8bit": 8_000_000, "qlora_4bit": 8_000_000}[compression]
                vram = base_vram_gb[compression] + rng.uniform(-0.1, 0.1)  # alpha와 무관한 잡음만
                latency = per_round_latency_sec[compression] * rounds + rng.uniform(-20, 20)
                results.append({
                    "compression": compression, "fl": fl, "alpha": alpha,
                    "val_perplexity": round(ppl, 4),
                    "rouge_l": round(rouge, 4),
                    "total_communication_bytes": payload_per_round * rounds * 8,
                    "rounds_run": rounds,
                    "peak_vram_gb": round(vram, 3),
                    "total_latency_sec": round(max(latency, 1.0), 1),
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
        # alpha는 10배씩 커지는 등비수열(0.1/1/10/100)이라 선형 축에 놓으면
        # 작은 값 셋이 한쪽 끝에 뭉쳐 눈금 라벨까지 겹친다. 도표 가독성만을
        # 위한 축 설정이며, compute_compression_alpha_trend의 상관계산은
        # 원본 alpha 그대로 유지한다(지도교수 확인).
        # alpha is geometric (0.1/1/10/100, x10 each step), so on a linear axis
        # the three small values collapse into one edge and even the tick labels
        # overlap. This is a chart-readability setting only — the correlation in
        # compute_compression_alpha_trend still uses raw alpha (advisor-confirmed).
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ax.minorticks_off()
        ax.invert_xaxis()  # 왼쪽(near-IID, alpha=100) -> 오른쪽(강한 non-IID, alpha=0.1)
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
        f.write(f"# {N_COMBOS}조합 결과 요약" + (" (SAMPLE DATA)" if is_sample else "") + "\n\n")
        f.write("| compression | fl | alpha | val_perplexity | rouge_l | rounds_run | total_communication_bytes | peak_vram_gb | total_latency_sec |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for compression in COMPRESSIONS:
            for fl in FLS:
                for alpha in ALPHAS:
                    r = next(x for x in results if x["compression"] == compression and x["fl"] == fl and x["alpha"] == alpha)
                    vram = r.get("peak_vram_gb")
                    vram_str = f"{vram:.3f}" if vram is not None else "N/A"
                    f.write(f"| {compression} | {fl} | {alpha} | {r['val_perplexity']:.4f} | "
                            f"{r['rouge_l']:.4f} | {r['rounds_run']} | {r['total_communication_bytes']:,} | "
                            f"{vram_str} | {r['total_latency_sec']:.1f} |\n")
    print(f"summary table -> {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    results = load_real_results()
    is_sample = results is None
    if is_sample:
        print(f"results/logs/에 {N_COMBOS}개 로그가 아직 없어 샘플 데이터로 미리보기를 생성합니다.")
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
    plot_trend(results, "peak_vram_gb", "Peak VRAM (GB, ↓ better)",
               "Peak VRAM vs. Dirichlet α, by compression level — expect ~flat lines",
               "peak_vram_vs_alpha.png", is_sample)
    plot_trend(results, "total_latency_sec", "Total Training Latency (sec, ↓ better)",
               "Total latency vs. Dirichlet α, by compression level", "latency_vs_alpha.png", is_sample)
    write_summary_table(results, is_sample)

    print(f"\n6 figures + 1 summary table written to {FIG_DIR}/")
    if is_sample:
        print("실제 GPU 실행 결과가 results/logs/에 쌓이면 이 스크립트를 다시 돌리세요 — 자동으로 진짜 데이터로 교체됩니다.")
        print("(Once real GPU results land in results/logs/, re-run this script — it will automatically switch to the real data.)")


if __name__ == "__main__":
    main()
