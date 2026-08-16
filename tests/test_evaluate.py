"""Subtask 2.3 분석 함수 + Subtask 1.3/2.1/2.2 상호작용/선정 로직 단위 테스트.

Unit tests for the Subtask 2.3 analysis functions and the Subtask 1.3/2.1/2.2
interaction/selection logic.
"""

from src.evaluate import (
    client_fairness_variance,
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
