"""Subtask 2.3 분석 함수 + Subtask 1.3/2.1/2.2 상호작용/선정 로직 단위 테스트.

Unit tests for the Subtask 2.3 analysis functions and the Subtask 1.3/2.1/2.2
interaction/selection logic.
"""

from src.evaluate import (
    client_fairness_variance,
    compute_compression_alpha_trend,
    compute_interaction_effects,
    per_category_breakdown,
    per_category_compression_penalty,
    score_generations_by_category,
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


def _make_18_combo_runs(penalties=(3.0, 2.5, 0.5)):
    """지도교수 피드백 반영: 3압축 x 2FL x 3alpha 가상 데이터. alpha가
    작을수록(non-IID가 강할수록) 압축 페널티가 커지도록 설계 —
    penalties는 alpha=[0.1, 1, 10] 순서의 (4bit-무압축) 성능차.

    Advisor feedback synthetic data: 3 compression x 2 FL x 3 alpha.
    Designed so the compression penalty grows as alpha decreases
    (stronger non-IID) — penalties are the (4-bit minus uncompressed)
    performance gap in alpha=[0.1, 1, 10] order.
    """
    runs = []
    for alpha, penalty in zip((0.1, 1, 10), penalties):
        for fl in ("fedavg", "fedprox"):
            runs.append({"compression": "lora", "fl": fl, "alpha": alpha, "val_perplexity": 10.0})
            runs.append({"compression": "qlora_8bit", "fl": fl, "alpha": alpha, "val_perplexity": 10.0 + penalty / 2})
            runs.append({"compression": "qlora_4bit", "fl": fl, "alpha": alpha, "val_perplexity": 10.0 + penalty})
    return runs


def test_compute_compression_alpha_trend_penalty_values():
    runs = _make_18_combo_runs(penalties=(3.0, 2.5, 0.5))
    trend = compute_compression_alpha_trend(runs, performance_field="val_perplexity")

    assert trend["fedavg"]["alphas"] == [0.1, 1, 10]
    assert trend["fedavg"]["compression_penalty"] == [3.0, 2.5, 0.5]
    assert trend["fedprox"]["compression_penalty"] == [3.0, 2.5, 0.5]


def test_compute_compression_alpha_trend_negative_correlation_when_penalty_grows_as_alpha_shrinks():
    # non-IID가 강할수록(alpha 작을수록) 압축 페널티가 커지도록 설계했으므로
    # alpha와 페널티는 음의 상관관계를 가져야 함.
    # Designed so the compression penalty grows as non-IID strengthens
    # (alpha shrinks), so alpha and penalty should be negatively correlated.
    runs = _make_18_combo_runs(penalties=(3.0, 2.5, 0.5))
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


def test_compute_compression_alpha_trend_compares_qlora_8bit_separately():
    """compression='qlora_8bit'을 넘기면 qlora_4bit이 아니라 8bit과 lora를
    비교해야 함 — _make_18_combo_runs는 8bit 페널티가 항상 4bit의 절반이
    되도록 설계돼 있으므로, 두 트렌드가 나란히 비교 가능해야 한다.

    Passing compression='qlora_8bit' must compare 8-bit against lora, not
    4-bit — _make_18_combo_runs is designed so the 8-bit penalty is always
    half the 4-bit penalty, so the two trends should be directly
    comparable."""
    runs = _make_18_combo_runs(penalties=(3.0, 2.5, 0.5))
    trend_4bit = compute_compression_alpha_trend(runs, performance_field="val_perplexity", compression="qlora_4bit")
    trend_8bit = compute_compression_alpha_trend(runs, performance_field="val_perplexity", compression="qlora_8bit")

    assert trend_4bit["fedavg"]["compression_penalty"] == [3.0, 2.5, 0.5]
    assert trend_8bit["fedavg"]["compression_penalty"] == [1.5, 1.25, 0.25]
    # 8bit도 4bit과 동일한 패턴(non-IID가 강할수록 페널티 증가)이어야 함
    # 8-bit should show the same pattern as 4-bit (penalty grows with non-IID)
    assert trend_8bit["fedavg"]["alpha_penalty_correlation"] < -0.9


def _make_18_combo_runs_rouge(penalties=(0.30, 0.15, 0.05)):
    """ROUGE-L(높을수록 좋음) 버전 — lora rouge_l=0.5 고정, qlora는 그보다
    정확히 penalty만큼 낮은 rouge_l을 갖도록 구성 (penalty가 클수록 압축
    피해가 큼).

    ROUGE-L (higher-is-better) version — lora's rouge_l is fixed at 0.5,
    qlora's rouge_l is exactly `penalty` lower (larger penalty = worse
    compression hit)."""
    runs = []
    for alpha, penalty in zip((0.1, 1, 10), penalties):
        for fl in ("fedavg", "fedprox"):
            runs.append({"compression": "lora", "fl": fl, "alpha": alpha, "rouge_l": 0.5})
            runs.append({"compression": "qlora_8bit", "fl": fl, "alpha": alpha, "rouge_l": 0.5 - penalty / 2})
            runs.append({"compression": "qlora_4bit", "fl": fl, "alpha": alpha, "rouge_l": 0.5 - penalty})
    return runs


def test_compute_compression_alpha_trend_higher_is_better_keeps_penalty_sign_correct():
    """ROUGE-L처럼 높을수록 좋은 지표를 higher_is_better=True로 넘기면,
    PPL(낮을수록 좋음)과 동일하게 "penalty>0 = 압축이 성능을 해쳤다"는
    의미가 유지돼야 함 — 부호를 반대로 읽는 실수를 막기 위한 회귀 테스트.

    Passing higher_is_better=True for a higher-is-better metric like
    ROUGE-L must preserve the same "penalty > 0 = compression hurt" meaning
    as the PPL (lower-is-better) case — a regression test against reading
    the sign backwards."""
    runs = _make_18_combo_runs_rouge(penalties=(0.30, 0.15, 0.05))
    trend = compute_compression_alpha_trend(runs, performance_field="rouge_l", higher_is_better=True)

    penalties = trend["fedavg"]["compression_penalty"]
    assert penalties[0] > penalties[1] > penalties[2] > 0
    # PPL 케이스와 동일하게, alpha가 작을수록 페널티가 커지므로 음의 상관관계여야 함
    assert trend["fedavg"]["alpha_penalty_correlation"] < -0.7
    assert trend["fedprox"]["alpha_penalty_correlation"] < -0.7


def test_compute_compression_alpha_trend_without_higher_is_better_flips_rouge_l_sign():
    """대조군: higher_is_better를 안 넘기면(기본값 False) ROUGE-L의 penalty가
    음수로 나와 "압축이 성능을 해쳤다"는 의미가 뒤집힌다는 것을 명시적으로
    확인 — higher_is_better 인자가 왜 필요한지 보여주는 회귀 테스트.

    Control: confirms that omitting higher_is_better (default False) for
    ROUGE-L produces a negative penalty — inverting the "positive =
    compression hurt" meaning. Documents why the higher_is_better argument
    is needed."""
    runs = _make_18_combo_runs_rouge(penalties=(0.30, 0.15, 0.05))
    trend = compute_compression_alpha_trend(runs, performance_field="rouge_l")
    assert trend["fedavg"]["compression_penalty"][0] < 0


def _make_category_runs():
    """카테고리별 압축×non-IID 상호작용 취약도 테스트용 가상 데이터.
    creative_writing은 alpha가 작을수록(non-IID 강할수록) rouge_l이 크게
    깎이도록(상대 페널티 0.60/0.20/0.04), closed_qa/general_qa는 alpha와
    무관하게 작고 일정한 페널티(0.03/0.05)를 갖도록 설계 — creative_writing만
    "취약 카테고리"로 플래그돼야 함.

    Synthetic data for testing per-category vulnerability to the
    compression x non-IID interaction. creative_writing is designed so
    rouge_l drops sharply as alpha shrinks (relative penalty 0.60/0.20/0.04);
    closed_qa/general_qa have small, alpha-independent penalties (0.03/0.05)
    — only creative_writing should be flagged "vulnerable"."""
    lora_per_cat = {
        "creative_writing": {"rouge_l": 0.5, "n": 20},
        "closed_qa": {"rouge_l": 0.5, "n": 20},
        "general_qa": {"rouge_l": 0.5, "n": 20},
    }
    qlora_per_cat_by_alpha = {
        0.1: {
            "creative_writing": {"rouge_l": 0.20, "n": 20},
            "closed_qa": {"rouge_l": 0.485, "n": 20},
            "general_qa": {"rouge_l": 0.475, "n": 20},
        },
        1: {
            "creative_writing": {"rouge_l": 0.40, "n": 20},
            "closed_qa": {"rouge_l": 0.485, "n": 20},
            "general_qa": {"rouge_l": 0.475, "n": 20},
        },
        10: {
            "creative_writing": {"rouge_l": 0.48, "n": 20},
            "closed_qa": {"rouge_l": 0.485, "n": 20},
            "general_qa": {"rouge_l": 0.475, "n": 20},
        },
    }
    runs = []
    for alpha in (0.1, 1, 10):
        runs.append({"fl": "fedavg", "alpha": alpha, "compression": "lora", "per_category": lora_per_cat})
        runs.append({"fl": "fedavg", "alpha": alpha, "compression": "qlora_4bit", "per_category": qlora_per_cat_by_alpha[alpha]})
    return runs


def test_per_category_compression_penalty_flags_vulnerable_category():
    runs = _make_category_runs()
    result = per_category_compression_penalty(runs, compression="qlora_4bit", metric="rouge_l", higher_is_better=True)

    assert result["creative_writing"]["vulnerable"] is True
    assert result["closed_qa"]["vulnerable"] is False
    assert result["general_qa"]["vulnerable"] is False
    assert result["creative_writing"]["mean_relative_penalty"] > result["closed_qa"]["mean_relative_penalty"]
    assert result["creative_writing"]["mean_relative_penalty"] > result["general_qa"]["mean_relative_penalty"]


def test_per_category_compression_penalty_alpha_correlation_matches_vulnerability():
    """취약 카테고리(creative_writing)는 페널티가 alpha가 작을수록 커지도록
    설계됐으니 강한 음의 상관관계를, 안정적인 카테고리들은 alpha와
    무관(상관계수 0)함을 확인.

    The vulnerable category (creative_writing) was designed so its penalty
    grows as alpha shrinks, so it should show a strong negative
    correlation; the stable categories should be alpha-independent
    (correlation 0)."""
    runs = _make_category_runs()
    result = per_category_compression_penalty(runs, compression="qlora_4bit", metric="rouge_l", higher_is_better=True)

    assert result["creative_writing"]["alpha_penalty_correlation"]["fedavg"] < -0.7
    assert result["closed_qa"]["alpha_penalty_correlation"]["fedavg"] == 0.0
    assert result["general_qa"]["alpha_penalty_correlation"]["fedavg"] == 0.0


def test_per_category_compression_penalty_reports_sample_size():
    runs = _make_category_runs()
    result = per_category_compression_penalty(runs, compression="qlora_4bit", metric="rouge_l", higher_is_better=True)
    assert result["creative_writing"]["n"]["fedavg_a0.1"] == 20


# ── score_generations_by_category: generations.jsonl -> per_category 입력 ──
# ── score_generations_by_category: generations.jsonl -> per_category input ──

def test_score_generations_by_category_averages_per_category():
    generations = [
        {"prediction": "The cat sat on the mat", "reference": "The cat sat on the mat", "category": "open_qa"},
        {"prediction": "completely unrelated text", "reference": "totally different sentence", "category": "open_qa"},
        {"prediction": "exact match", "reference": "exact match", "category": "closed_qa"},
    ]
    result = score_generations_by_category(generations)

    assert result["closed_qa"]["n"] == 1
    assert result["closed_qa"]["rouge_l"] > 0.99
    assert result["open_qa"]["n"] == 2
    # 완전일치 1개 + 무관한 1개의 평균이라 0과 1 사이 중간값이어야 함
    assert 0.0 < result["open_qa"]["rouge_l"] < 1.0


def test_score_generations_by_category_output_feeds_per_category_compression_penalty():
    """score_generations_by_category의 출력 shape가 per_category_compression_penalty가
    기대하는 {category: {"rouge_l":.., "n":..}} 형태와 실제로 맞물리는지 확인
    (두 함수가 항상 같이 쓰이므로 계약이 안 맞으면 즉시 깨져야 함).

    Confirms score_generations_by_category's output shape actually matches
    what per_category_compression_penalty expects
    ({category: {"rouge_l":.., "n":..}}) — since the two are always used
    together, a contract mismatch should fail immediately."""
    lora_gen = [{"prediction": "a", "reference": "a", "category": "open_qa"}]
    qlora_gen = [{"prediction": "totally unrelated", "reference": "a", "category": "open_qa"}]

    runs = [
        {"fl": "fedavg", "alpha": 0.1, "compression": "lora", "per_category": score_generations_by_category(lora_gen)},
        {"fl": "fedavg", "alpha": 0.1, "compression": "qlora_4bit", "per_category": score_generations_by_category(qlora_gen)},
    ]
    result = per_category_compression_penalty(runs, compression="qlora_4bit", metric="rouge_l", higher_is_better=True)
    assert "open_qa" in result
    assert result["open_qa"]["mean_relative_penalty"] > 0
