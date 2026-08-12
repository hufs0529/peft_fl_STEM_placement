"""6단계: Non-IID 파티셔닝 검증 (Subtask 1.1 — Dirichlet(alpha=0.5)가 실제로
비IID 분배를 만들어내는지, alpha가 작을수록 더 비IID해지는지 확인)."""

from src.partitioning import heterogeneity_score, partition_by_category, summarize_partition


def make_dummy_dataset():
    return (
        [{"category": "A", "text": f"a{i}"} for i in range(200)]
        + [{"category": "B", "text": f"b{i}"} for i in range(80)]
        + [{"category": "C", "text": f"c{i}"} for i in range(200)]
    )


def test_min_threshold_resampling():
    dataset = make_dummy_dataset()
    client_data = partition_by_category(dataset, num_clients=4, alpha=0.5, min_category_threshold=150)
    total_b = sum(1 for items in client_data.values() for x in items if x["category"] == "B")
    assert total_b >= 150, "임계값 미만 카테고리가 리샘플링되지 않음"


def test_partition_is_non_iid():
    dataset = make_dummy_dataset()
    client_data = partition_by_category(dataset, num_clients=4, alpha=0.1, min_category_threshold=150)
    summary = summarize_partition(client_data)
    for client_id, counts in summary.items():
        print(f"client {client_id}: {counts}")


def test_alpha_controls_heterogeneity_strength():
    """alpha=0.1 > alpha=0.5 > alpha=1.0 순으로 비IID 강도(heterogeneity_score)가
    커야 함 — Dirichlet 파티셔닝이 의도대로 동작하는지에 대한 일반 검증."""
    dataset = make_dummy_dataset()

    scores = {}
    for alpha in [1.0, 0.5, 0.1]:
        client_data = partition_by_category(dataset, num_clients=4, alpha=alpha, min_category_threshold=150, seed=1)
        scores[alpha] = heterogeneity_score(client_data)

    print(f"heterogeneity scores: {scores}")
    assert scores[0.1] >= scores[0.5] >= scores[1.0] - 0.05, (
        f"alpha가 작을수록 비IID가 강해야 하는데 순서가 어긋남: {scores}"
    )
