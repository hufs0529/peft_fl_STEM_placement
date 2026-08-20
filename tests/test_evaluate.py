"""Subtask 2.3 분석 함수 + Subtask 1.3/2.1/2.2 상호작용/선정 로직 단위 테스트.

Unit tests for the Subtask 2.3 analysis functions and the Subtask 1.3/2.1/2.2
interaction/selection logic.
"""

from src.evaluate import (
    client_fairness_variance,
    compute_compression_alpha_trend,
    compute_interaction_effects,
    per_category_breakdown,
    select_best_performing_combination,
    select_largest_interaction_fl_algorithm,
    total_communication_cost,
)


def test_per_category_breakdown():
    results = [
        {"category": "open_qa", "ppl": 10.0, "rouge_l": 0.5},
        {"category": "open_qa", "ppl": 12.0, "rouge_l": 0.4},
        {"category": "creative_writing", "ppl": 20.0, "rouge_l": 0.2},
    ]
    breakdown = per_category_breakdown(results)
    assert breakdown["open_qa"]["ppl"] == 11.0
    assert breakdown["open_qa"]["n"] == 2
    assert breakdown["creative_writing"]["n"] == 1


def test_client_fairness_variance_zero_when_equal():
    assert client_fairness_variance({0: 5.0, 1: 5.0, 2: 5.0}) == 0.0


def test_client_fairness_variance_positive_when_unequal():
    assert client_fairness_variance({0: 1.0, 1: 10.0}) > 0


def test_total_communication_cost():
    assert total_communication_cost(payload_bytes_per_round=1000, rounds_to_converge=5) == 5000


def _make_core_runs(fedprox_ppl_qlora=9.5, scaffold_ppl_qlora=9.0):
    """lora에서는 fedprox/scaffold가 fedavg보다 뚜렷이 개선되지만,
    qlora에서는 fedprox의 개선폭이 줄어드는(상호작용이 있는) 가상 6조합.

    Synthetic set of 6 combinations where fedprox/scaffold clearly improve
    over fedavg under lora, but fedprox's improvement shrinks under qlora
    (i.e., an interaction effect is present).
    """
    return [
        {"peft": "lora", "fl": "fedavg", "val_perplexity": 10.0},
        {"peft": "lora", "fl": "fedprox", "val_perplexity": 8.0},
        {"peft": "lora", "fl": "scaffold", "val_perplexity": 7.5},
        {"peft": "qlora", "fl": "fedavg", "val_perplexity": 10.5},
        {"peft": "qlora", "fl": "fedprox", "val_perplexity": fedprox_ppl_qlora},
        {"peft": "qlora", "fl": "scaffold", "val_perplexity": scaffold_ppl_qlora},
    ]


def test_compute_interaction_effects_sign_and_shape():
    runs = _make_core_runs(fedprox_ppl_qlora=9.5, scaffold_ppl_qlora=9.0)
    effects = compute_interaction_effects(runs, performance_field="val_perplexity")

    # lora: fedprox_delta = 10.0 - 8.0 = 2.0 (개선)
    # lora: fedprox_delta = 10.0 - 8.0 = 2.0 (improvement)
    assert effects["per_peft"]["lora"]["fedprox_delta"] == 2.0
    # qlora: fedprox_delta = 10.5 - 9.5 = 1.0 (개선폭이 줄어듦 -> 상호작용 존재)
    # qlora: fedprox_delta = 10.5 - 9.5 = 1.0 (improvement shrinks -> interaction exists)
    assert effects["per_peft"]["qlora"]["fedprox_delta"] == 1.0
    assert effects["interaction_fedprox"] == 1.0 - 2.0


def test_select_largest_interaction_fl_algorithm_picks_bigger_magnitude():
    # fedprox 상호작용(-1.0)보다 scaffold 상호작용(-2.5)이 더 크게 설계
    # Designed so the scaffold interaction (-2.5) is larger in magnitude
    # than the fedprox interaction (-1.0)
    runs = _make_core_runs(fedprox_ppl_qlora=9.5, scaffold_ppl_qlora=10.0)
    effects = compute_interaction_effects(runs, performance_field="val_perplexity")
    assert select_largest_interaction_fl_algorithm(effects) == "scaffold"


def test_select_best_performing_combination_prefers_lowest_perplexity():
    runs = _make_core_runs()
    best = select_best_performing_combination(runs, performance_field="val_perplexity")
    assert best["peft"] == "lora" and best["fl"] == "scaffold"


def _make_18_combo_runs(penalties=(3.0, 1.5, 0.5)):
    """지도교수 피드백 반영: 3압축 x 2FL x 3alpha 가상 데이터. alpha가
    작을수록(non-IID가 강할수록) 압축 페널티가 커지도록 설계 —
    penalties는 alpha=[0.1, 0.5, 1.0] 순서의 (4bit-무압축) 성능차.

    Advisor feedback synthetic data: 3 compression x 2 FL x 3 alpha.
    Designed so the compression penalty grows as alpha decreases
    (stronger non-IID) — penalties are the (4-bit minus uncompressed)
    performance gap in alpha=[0.1, 0.5, 1.0] order.
    """
    runs = []
    for alpha, penalty in zip((0.1, 0.5, 1.0), penalties):
        for fl in ("fedavg", "fedprox"):
            runs.append({"compression": "lora", "fl": fl, "alpha": alpha, "val_perplexity": 10.0})
            runs.append({"compression": "qlora_8bit", "fl": fl, "alpha": alpha, "val_perplexity": 10.0 + penalty / 2})
            runs.append({"compression": "qlora_4bit", "fl": fl, "alpha": alpha, "val_perplexity": 10.0 + penalty})
    return runs


def test_compute_compression_alpha_trend_penalty_values():
    runs = _make_18_combo_runs(penalties=(3.0, 1.5, 0.5))
    trend = compute_compression_alpha_trend(runs, performance_field="val_perplexity")

    assert trend["fedavg"]["alphas"] == [0.1, 0.5, 1.0]
    assert trend["fedavg"]["compression_penalty"] == [3.0, 1.5, 0.5]
    assert trend["fedprox"]["compression_penalty"] == [3.0, 1.5, 0.5]


def test_compute_compression_alpha_trend_negative_correlation_when_penalty_grows_as_alpha_shrinks():
    # non-IID가 강할수록(alpha 작을수록) 압축 페널티가 커지도록 설계했으므로
    # alpha와 페널티는 음의 상관관계를 가져야 함.
    # Designed so the compression penalty grows as non-IID strengthens
    # (alpha shrinks), so alpha and penalty should be negatively correlated.
    runs = _make_18_combo_runs(penalties=(3.0, 1.5, 0.5))
    trend = compute_compression_alpha_trend(runs, performance_field="val_perplexity")
    assert trend["fedavg"]["alpha_penalty_correlation"] < -0.9
    assert trend["fedprox"]["alpha_penalty_correlation"] < -0.9


def test_compute_compression_alpha_trend_near_zero_correlation_when_penalty_constant():
    # 압축 페널티가 alpha와 무관하게 항상 동일하면 상관관계는 0에 가까워야 함.
    # If the compression penalty is constant regardless of alpha, the
    # correlation should be near zero.
    runs = _make_18_combo_runs(penalties=(2.0, 2.0, 2.0))
    trend = compute_compression_alpha_trend(runs, performance_field="val_perplexity")
    assert trend["fedavg"]["alpha_penalty_correlation"] == 0.0
