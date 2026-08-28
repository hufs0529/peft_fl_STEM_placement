"""추가 GPU 비용 없이 이미 로깅된 데이터에서 계산하는 분석 함수들.

- per_category_breakdown / total_communication_cost: Subtask 2.3 진단
  분석(카테고리별 분해, 통신비용)용 일반 유틸리티.
- client_fairness_variance: 클라이언트별 성능 분산을 계산하는 순수 함수.
  현재 fl_runner.py의 held-out set은 전역 공유(클라이언트마다 다르지 않음)
  방식이라 파이프라인에서는 호출하지 않는다 — 값이 항상 동일해 의미가
  없기 때문(README 알려진 한계 참고). 클라이언트별로 분포가 다른 eval
  set을 나중에 도입할 경우를 위해 유틸리티로만 남겨둔다.
- compute_interaction_effects: Subtask 1.3 — Aim 1의 직접적인 정량적 답.
  performance_field로 val_perplexity(성능)뿐 아니라 rounds_run(수렴 속도)도
  그대로 넘길 수 있다 (scripts/analyze_interaction.py 참고).
- select_largest_interaction_fl_algorithm: Subtask 2.1에서 DoRA와 짝지을
  FL 알고리즘(상호작용 효과가 가장 컸던 쪽)을 고른다.
- select_best_performing_combination: Subtask 2.2 local-epoch=5 강건성
  점검에 쓸 "core 6조합 중 성능이 가장 좋은 조합"을 고른다.
- compute_compression_alpha_trend: 지도교수 피드백(압축률 x Dirichlet
  강건성 스윕) — 3압축(lora/qlora_8bit/qlora_4bit) x 2FL(fedavg/fedprox)
  x 3alpha(0.1/1/10) 18조합에서, non-IID가 강해질수록(alpha가
  작을수록) 압축 페널티가 커지는지를 Pearson 상관계수로 정량화한다.

Analysis functions computed from data that is already logged, at no extra
GPU cost.

- per_category_breakdown / total_communication_cost: general-purpose
  utilities for Subtask 2.3 diagnostic analysis (per-category breakdown,
  communication cost).
- client_fairness_variance: a pure function that computes the variance of
  per-client performance. Currently the held-out set in fl_runner.py is
  shared globally (it does not differ per client), so the pipeline does
  not call this — the value would always be identical and thus meaningless
  (see the README's known limitations). It is kept as a utility for a
  future case where a per-client eval set with a differing distribution
  is introduced.
- compute_interaction_effects: Subtask 1.3 — the direct quantitative
  answer to Aim 1. The performance_field argument accepts not only
  val_perplexity (performance) but also rounds_run (convergence speed)
  as-is (see scripts/analyze_interaction.py).
- select_largest_interaction_fl_algorithm: picks the FL algorithm to pair
  with DoRA in Subtask 2.1 (the one with the larger interaction effect).
- select_best_performing_combination: picks "the best-performing
  combination among the core 6 combinations" to use for the Subtask 2.2
  local-epoch=5 robustness check.
- compute_compression_alpha_trend: advisor feedback (compression-rate x
  Dirichlet robustness sweep) — over the 18 combinations formed by 3
  compression levels (lora/qlora_8bit/qlora_4bit) x 2 FL algorithms
  (fedavg/fedprox) x 3 alpha values (0.1/1/10), quantifies via a
  Pearson correlation coefficient whether the compression penalty grows
  as non-IID intensity increases (alpha decreases).
"""

from collections import defaultdict
from typing import Dict, List


def per_category_breakdown(per_example_results: List[dict]) -> Dict[str, dict]:
    """카테고리별 PPL/ROUGE-L 평균 — 압축 아티팩트가 특정 태스크
    (예: creative_writing)에 집중되는지 확인 (Subtask 2.3).

    Per-category average PPL/ROUGE-L — checks whether compression
    artifacts concentrate on a specific task (e.g. creative_writing)
    (Subtask 2.3)."""
    grouped = defaultdict(list)
    for r in per_example_results:
        grouped[r["category"]].append(r)

    return {
        cat: {
            "ppl": sum(x["ppl"] for x in items) / len(items),
            "rouge_l": sum(x["rouge_l"] for x in items) / len(items),
            "n": len(items),
        }
        for cat, items in grouped.items()
    }


def client_fairness_variance(per_client_metrics: Dict[int, float]) -> float:
    """클라이언트별 성능의 분산 — 교정 알고리즘이 클라이언트 간
    불균형을 줄이는지(평균만으로는 안 보이는 공정성) 확인.

    Variance of per-client performance — checks whether the correction
    algorithm reduces cross-client imbalance (fairness that a mean alone
    cannot reveal)."""
    values = list(per_client_metrics.values())
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def total_communication_cost(payload_bytes_per_round: int, rounds_to_converge: int) -> int:
    """payload x R — Subtask 1.2에서 요구하는 communication cost 지표.

    payload x R — the communication cost metric required by Subtask 1.2."""
    return payload_bytes_per_round * rounds_to_converge


def compute_interaction_effects(run_results: List[dict], performance_field: str = "val_perplexity") -> Dict[str, dict]:
    """Subtask 1.3: 3x2 factorial(core 6조합)의 상호작용 효과 계산.

    각 PEFT(lora/qlora)에 대해 FedProx/SCAFFOLD가 FedAvg 대비 얼마나
    성능을 개선하는지(delta, performance_field는 낮을수록 좋다고 가정하므로
    delta = fedavg - other, 양수면 개선)를 구하고, 그 delta가 LoRA에서
    QLoRA로 갈 때 얼마나 달라지는지(interaction = delta_qlora - delta_lora)를
    계산한다. 이것이 "교정 정교함의 이점이 양자화와 결합해도 유지되는가"에
    대한 직접적인 정량적 답이다.

    run_results: [{"peft": "lora"|"qlora", "fl": "fedavg"|"fedprox"|"scaffold",
                    performance_field: float, ...}, ...] (6개 core 조합)

    Subtask 1.3: computes the interaction effect of a 3x2 factorial
    (the core 6 combinations).

    For each PEFT (lora/qlora), computes how much FedProx/SCAFFOLD improve
    performance relative to FedAvg (delta; performance_field is assumed to
    be lower-is-better, so delta = fedavg - other, positive means
    improvement), then computes how much that delta changes going from
    LoRA to QLoRA (interaction = delta_qlora - delta_lora). This is the
    direct quantitative answer to "does the benefit of correction
    sophistication persist when combined with quantization?"

    run_results: [{"peft": "lora"|"qlora", "fl": "fedavg"|"fedprox"|"scaffold",
                    performance_field: float, ...}, ...] (the 6 core combinations)
    """
    by_combo = {(r["peft"], r["fl"]): r[performance_field] for r in run_results}

    effects: Dict[str, dict] = {}
    for peft in ("lora", "qlora"):
        fedavg = by_combo[(peft, "fedavg")]
        effects[peft] = {
            "fedavg": fedavg,
            "fedprox_delta": fedavg - by_combo[(peft, "fedprox")],
            "scaffold_delta": fedavg - by_combo[(peft, "scaffold")],
        }

    return {
        "per_peft": effects,
        "interaction_fedprox": effects["qlora"]["fedprox_delta"] - effects["lora"]["fedprox_delta"],
        "interaction_scaffold": effects["qlora"]["scaffold_delta"] - effects["lora"]["scaffold_delta"],
    }


def select_largest_interaction_fl_algorithm(interaction_effects: Dict[str, dict]) -> str:
    """Subtask 2.1: DoRA 진단 실험과 짝지을 FL 알고리즘을 고른다 —
    Task 1에서 |interaction|이 가장 컸던 쪽(양자화 노이즈에 가장 민감했던
    교정 메커니즘).

    Subtask 2.1: picks the FL algorithm to pair with the DoRA diagnostic
    experiment — whichever had the larger |interaction| in Task 1 (the
    correction mechanism most sensitive to quantization noise)."""
    candidates = {
        "fedprox": abs(interaction_effects["interaction_fedprox"]),
        "scaffold": abs(interaction_effects["interaction_scaffold"]),
    }
    return max(candidates, key=candidates.get)


def select_best_performing_combination(run_results: List[dict], performance_field: str = "val_perplexity") -> dict:
    """Subtask 2.2: local_epochs=5 강건성 점검에 쓸 "core 6조합 중
    성능이 가장 좋은 조합"을 고른다 (performance_field는 낮을수록 좋음).
    통신비용은 고려하지 않는다 — 순수 task performance 기준.

    Subtask 2.2: picks "the best-performing combination among the core 6
    combinations" to use for the local_epochs=5 robustness check
    (performance_field is lower-is-better). Communication cost is not
    considered — this is based purely on task performance."""
    return min(run_results, key=lambda r: r[performance_field])


def compute_compression_alpha_trend(
    run_results: List[dict], performance_field: str = "val_perplexity"
) -> Dict[str, dict]:
    """지도교수 피드백: 압축률(compression) x Dirichlet 비IID 강도(alpha)
    상관관계 분석.

    각 (fl, alpha) 지점에서 "압축 페널티" = QLoRA-4bit 성능 - LoRA(무압축)
    성능을 구한다(performance_field는 낮을수록 좋다고 가정하므로, 양수면
    압축이 성능을 해쳤다는 뜻). alpha가 작아질수록(non-IID가 강해질수록)
    이 페널티가 커지는 경향이 있는지 Pearson 상관계수로 확인한다.

    상관계수가 음수: alpha가 작을수록(non-IID가 강할수록) 페널티가
    커짐 -> 압축과 non-IID가 서로를 증폭시킴(악화 방향의 상호작용).
    0에 가까움: 압축 페널티가 non-IID 강도와 무관 -> 두 축이 독립적.

    run_results: [{"compression": "lora"|"qlora_8bit"|"qlora_4bit",
                    "fl": "fedavg"|"fedprox", "alpha": float,
                    performance_field: float, ...}, ...] (18개 조합)

    Advisor feedback: analyzes the correlation between compression rate
    and Dirichlet non-IID intensity (alpha).

    For each (fl, alpha) point, computes the "compression penalty" =
    QLoRA-4bit performance - LoRA (uncompressed) performance
    (performance_field is assumed lower-is-better, so a positive value
    means compression hurt performance). Checks via a Pearson correlation
    coefficient whether this penalty tends to grow as alpha decreases
    (non-IID intensity increases).

    Negative correlation: the penalty grows as alpha decreases (stronger
    non-IID) -> compression and non-IID amplify each other (an adverse
    interaction). Near zero: the compression penalty is independent of
    non-IID intensity -> the two axes are independent.

    run_results: [{"compression": "lora"|"qlora_8bit"|"qlora_4bit",
                    "fl": "fedavg"|"fedprox", "alpha": float,
                    performance_field: float, ...}, ...] (the 18 combinations)
    """
    import numpy as np

    by_key = {(r["fl"], r["alpha"], r["compression"]): r[performance_field] for r in run_results}
    alphas = sorted({r["alpha"] for r in run_results})
    fls = sorted({r["fl"] for r in run_results})

    trends: Dict[str, dict] = {}
    for fl in fls:
        penalties = [by_key[(fl, alpha, "qlora_4bit")] - by_key[(fl, alpha, "lora")] for alpha in alphas]
        # penalties가 전부 동일(분산 0)하면 numpy가 0으로 나누기 때문에
        # 상관계수가 NaN이 됨 -> "alpha와 무관하다"는 의미로 0.0을 명시적으로 반환.
        # If penalties have zero variance (all identical), numpy divides by
        # zero and the correlation becomes NaN -> explicitly return 0.0 to
        # mean "independent of alpha".
        if len(alphas) > 1 and len(set(penalties)) > 1:
            correlation = float(np.corrcoef(alphas, penalties)[0, 1])
        else:
            correlation = 0.0
        trends[fl] = {
            "alphas": alphas,
            "compression_penalty": penalties,
            "alpha_penalty_correlation": correlation,
        }

    return trends
