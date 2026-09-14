"""Week 5 분석 (지도교수 피드백 반영: 압축률 x Dirichlet 스윕, Subtask 1.3/2.1/2.2 선정)
— 24조합(3압축 x 2FL x 4alpha) 완료 후 실행.

3압축(lora/qlora_8bit/qlora_4bit) x 2FL(fedavg/fedprox) x 4alpha(0.1/1/10/100)
= 24개 실행 로그를 모두 읽어:
  1) 압축률 x Dirichlet alpha 격자표 출력
  2) 압축 페널티(4bit/8bit 각각 - 무압축)가 alpha가 작아질수록(non-IID가
     강해질수록) 커지는지 Pearson 상관계수로 확인, PPL/ROUGE-L(높을수록
     좋음, higher_is_better=True) 둘 다 (src/evaluate.py::compute_compression_alpha_trend)
  3) 카테고리별 압축×non-IID 취약도 스크리닝(Subtask 2.3) — 어떤 태스크
     카테고리가 압축×non-IID 상호작용에 특히 취약한지 z-score로 확인
     (src/evaluate.py::per_category_compression_penalty,
     `*_generations.jsonl`에서 예제별 ROUGE-L을 다시 계산)
  4) Subtask 2.2: local_epochs=5 강건성 점검에 쓸 "24조합 중 성능이 가장
     좋은 조합" 결정

SCAFFOLD/DoRA는 이 24조합 스코프에서 빠졌습니다 — scaffold.py 자체는
그대로 남아있고 테스트도 통과하지만, "core 실험 설계"에는 포함되지 않습니다
(README 참고).

사용법:
    python scripts/analyze_interaction.py
(results/logs/ 안의 24개 조합 *_rounds.jsonl을 모두 읽어 계산)

---

Week 5 analysis (reflecting advisor feedback: compression-rate x
Dirichlet sweep, Subtask 1.3/2.1/2.2 selection) — run once all 24
combinations (3 compression x 2 FL x 4 alpha) are complete.

Reads the logs of all 24 runs formed by 3 compression levels
(lora/qlora_8bit/qlora_4bit) x 2 FL algorithms (fedavg/fedprox) x 4 alpha
values (0.1/1/10/100) to:
  1) print a compression x Dirichlet-alpha grid,
  2) check via a Pearson correlation coefficient whether the compression
     penalty (4-bit and 8-bit, each minus uncompressed) grows as alpha
     decreases (non-IID intensity increases), for both PPL and ROUGE-L
     (higher-is-better, higher_is_better=True)
     (src/evaluate.py::compute_compression_alpha_trend),
  3) per-category compression x non-IID vulnerability screening
     (Subtask 2.3) — which task categories are especially vulnerable to
     the interaction, via a z-score
     (src/evaluate.py::per_category_compression_penalty, recomputing
     per-example ROUGE-L from `*_generations.jsonl`),
  4) Subtask 2.2: pick "the best-performing combination among the 24" to
     use for the local_epochs=5 robustness check.

SCAFFOLD/DoRA are out of scope for these 24 combinations — scaffold.py
itself still exists and its tests still pass, but it is not part of the
core experiment design (see the README).

Usage:
    python scripts/analyze_interaction.py
(reads all 24 combinations' *_rounds.jsonl files under results/logs/)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (
    compute_compression_alpha_trend,
    per_category_compression_penalty,
    score_generations_by_category,
    select_best_performing_combination,
)

COMPRESSIONS = ["lora", "qlora_8bit", "qlora_4bit"]
FLS = ["fedavg", "fedprox"]
# 지도교수 피드백(2026-09-08): near-IID 끝점으로 alpha=100 추가
# Advisor feedback (2026-09-08): added alpha=100 as the near-IID endpoint
ALPHAS = [0.1, 1, 10, 100]
N_COMBOS = len(COMPRESSIONS) * len(FLS) * len(ALPHAS)


def build_run_name(compression: str, fl: str, alpha: float) -> str:
    """scripts/run_experiment.py의 build_run_name()과 정확히 동일한 규칙.

    Exactly mirrors build_run_name() in scripts/run_experiment.py."""
    peft = "lora" if compression == "lora" else "qlora"
    name = f"{peft}_{fl}_ep1"
    if alpha != 1:
        # float()로 정규화: --alpha 10 -> "a10.0" (실제 로그 파일명과 일치)
        # Normalize via float(): --alpha 10 -> "a10.0", matching the log filenames
        name += f"_a{float(alpha)}"
    if compression == "qlora_8bit":
        name += "_q8bit"
    return name


def load_run_summary(compression: str, fl: str, alpha: float) -> Optional[dict]:
    run_name = build_run_name(compression, fl, alpha)
    path = f"results/logs/{run_name}_rounds.jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    last = lines[-1]
    total_comm = sum(r["communication_bytes_this_round"] for r in lines)
    return {
        "compression": compression,
        "fl": fl,
        "alpha": alpha,
        "run_name": run_name,
        "final_val_loss": last["val_loss"],
        "val_perplexity": last["val_perplexity"],
        "rouge_l": last.get("rouge_l"),
        "total_communication_bytes": total_comm,
        "rounds_run": len(lines),
    }


def load_run_per_category(compression: str, fl: str, alpha: float) -> Optional[dict]:
    """`{run_name}_generations.jsonl`에서 카테고리별 ROUGE-L을 계산한다.
    파일이 없으면(예: 그 run이 rouge_l을 하나도 못 만들었으면) None.

    Computes per-category ROUGE-L from `{run_name}_generations.jsonl`.
    Returns None if the file doesn't exist (e.g. that run produced no
    rouge_l at all)."""
    run_name = build_run_name(compression, fl, alpha)
    path = f"results/logs/{run_name}_generations.jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        generations = [json.loads(line) for line in f]
    return score_generations_by_category(generations)


def main():
    run_results: List[dict] = []
    missing = []
    for compression in COMPRESSIONS:
        for fl in FLS:
            for alpha in ALPHAS:
                summary = load_run_summary(compression, fl, alpha)
                if summary is None:
                    missing.append(build_run_name(compression, fl, alpha))
                    continue
                run_results.append(summary)

    if missing:
        print(f"\n경고: {len(missing)}/{N_COMBOS} 조합 로그를 찾지 못함 — 아직 실행되지 않았을 수 있음:")
        for name in missing:
            print(f"  - {name}")
        if len(run_results) < N_COMBOS:
            print(f"\n{len(run_results)}/{N_COMBOS} 조합만 완료됨. 전부 완료 후 다시 실행하세요.")
            return

    print(f"\n=== {N_COMBOS}조합 격자 (val_perplexity, 낮을수록 좋음) ===")
    header = f"{'compression':12s} {'fl':10s} " + " ".join(f"a={a:<6}" for a in ALPHAS)
    print(header)
    for compression in COMPRESSIONS:
        for fl in FLS:
            row = [r for r in run_results if r["compression"] == compression and r["fl"] == fl]
            row_by_alpha = {r["alpha"]: r["val_perplexity"] for r in row}
            cells = " ".join(f"{row_by_alpha.get(a, float('nan')):<8.4f}" for a in ALPHAS)
            print(f"{compression:12s} {fl:10s} {cells}")

    print("\n=== 압축률 x Dirichlet alpha 상관관계 (PPL 기준) ===")
    for compression in ("qlora_4bit", "qlora_8bit"):
        trend = compute_compression_alpha_trend(run_results, performance_field="val_perplexity", compression=compression)
        print(f"-- {compression} vs lora --")
        for fl, t in trend.items():
            print(f"  {fl}: alpha={t['alphas']}  압축_페널티({compression}-무압축)={[round(p, 4) for p in t['compression_penalty']]}")
            print(f"       alpha-페널티 상관계수 = {t['alpha_penalty_correlation']:+.4f}"
                  " (음수면 non-IID가 강할수록 압축 페널티가 커짐 = 상호작용 있음)")
    print("  (4bit 상관계수가 8bit보다 더 음수면: 압축이 강할수록 non-IID 민감도도 커지는 dose-response)")

    if all(r.get("rouge_l") is not None for r in run_results):
        print("\n=== 압축률 x Dirichlet alpha 상관관계 (ROUGE-L 기준, 높을수록 좋음) ===")
        for compression in ("qlora_4bit", "qlora_8bit"):
            trend = compute_compression_alpha_trend(
                run_results, performance_field="rouge_l", compression=compression, higher_is_better=True,
            )
            print(f"-- {compression} vs lora --")
            for fl, t in trend.items():
                print(f"  {fl}: alpha={t['alphas']}  압축_페널티({compression}-무압축)={[round(p, 4) for p in t['compression_penalty']]}")
                print(f"       alpha-페널티 상관계수 = {t['alpha_penalty_correlation']:+.4f}")
    else:
        print("\n(일부 run에 rouge_l이 없어 ROUGE-L 기준 상관분석은 건너뜀)")

    conv_trend = compute_compression_alpha_trend(run_results, performance_field="rounds_run")
    print("\n=== 압축률 x Dirichlet alpha 상관관계 (수렴 속도 rounds_run 기준, 완전 무료로 이미 로깅된 값) ===")
    for fl, t in conv_trend.items():
        print(f"  {fl}: alpha={t['alphas']}  압축_페널티(4bit-무압축)={[round(p, 4) for p in t['compression_penalty']]}")
        print(f"       alpha-페널티 상관계수 = {t['alpha_penalty_correlation']:+.4f}")

    print("\n=== 카테고리별 압축×non-IID 취약도 스크리닝 (ROUGE-L 기준, Subtask 2.3) ===")
    per_category_runs = []
    missing_gen = []
    for compression in COMPRESSIONS:
        for fl in FLS:
            for alpha in ALPHAS:
                per_cat = load_run_per_category(compression, fl, alpha)
                if per_cat is None:
                    missing_gen.append(build_run_name(compression, fl, alpha))
                    continue
                per_category_runs.append({"compression": compression, "fl": fl, "alpha": alpha, "per_category": per_cat})

    if len(per_category_runs) < N_COMBOS:
        print(f"  ({len(per_category_runs)}/{N_COMBOS}개 generations.jsonl만 존재 — 건너뜀: {missing_gen})")
    else:
        for compression in ("qlora_4bit", "qlora_8bit"):
            penalty = per_category_compression_penalty(
                per_category_runs, compression=compression, metric="rouge_l", higher_is_better=True,
            )
            print(f"-- {compression} vs lora (z-score 내림차순, z>1이면 취약) --")
            for cat, stats in sorted(penalty.items(), key=lambda kv: -kv[1]["z_score"]):
                flag = "  <== 취약" if stats["vulnerable"] else ""
                print(f"  {cat:20s} mean_relative_penalty={stats['mean_relative_penalty']:+.4f}  "
                      f"z={stats['z_score']:+.2f}{flag}")

    best = select_best_performing_combination(run_results, performance_field="val_perplexity")
    print(f"\n>>> Subtask 2.2: local_epochs=5 강건성 점검 대상 = {best['run_name']} <<<")
    peft_arg = "lora" if best["compression"] == "lora" else "qlora"
    bits_arg = "" if best["compression"] != "qlora_8bit" else " --qlora-bits 8"
    print(f"    python scripts/run_experiment.py --peft {peft_arg} --fl {best['fl']} "
          f"--alpha {best['alpha']}{bits_arg} --local-epochs 5")


if __name__ == "__main__":
    main()
