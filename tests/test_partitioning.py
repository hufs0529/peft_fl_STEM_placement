"""6단계: Non-IID 파티셔닝 검증 (Subtask 1.1 — Dirichlet(alpha=0.5)가 실제로
비IID 분배를 만들어내는지, alpha가 작을수록 더 비IID해지는지 확인).

더미 데이터 테스트(빠름, 네트워크 불필요)와 실제 Dolly-15k 전체(15,011개)
데이터 테스트(네트워크 필요, @pytest.mark.network)로 나뉜다. Dolly 데이터는
`dolly_dataset` fixture(module scope)로 파일당 1회만 로드해 재사용한다.
파티셔닝 자체는 numpy 기반 순수 연산이라 15,011개를 다뤄도 매우 빠르다 —
느린 건 최초 1회의 HuggingFace 다운로드뿐이며 이후엔 로컬 캐시를 쓴다.

기본 실행(`pytest tests/`)에서 네트워크 테스트를 빼려면:
    pytest tests/ -m "not network"

Step 6: Verify non-IID partitioning (Subtask 1.1 — confirm that
Dirichlet(alpha=0.5) actually produces a non-IID distribution, and that a
smaller alpha yields stronger non-IID).

Split into dummy-data tests (fast, no network) and real full Dolly-15k
(15,011 examples) tests (needs network, @pytest.mark.network). The Dolly
data is loaded once per file via a module-scoped `dolly_dataset` fixture.
Partitioning itself is pure numpy computation, so handling 15,011 rows is
very fast — the only slow part is the one-time HuggingFace download,
which is cached locally afterwards.

To skip network tests in a normal run (`pytest tests/`):
    pytest tests/ -m "not network"
"""

from collections import Counter
from typing import List

import pytest

from src.partitioning import heterogeneity_score, partition_by_category, summarize_partition


def make_dummy_dataset():
    return (
        [{"category": "A", "text": f"a{i}"} for i in range(200)]
        + [{"category": "B", "text": f"b{i}"} for i in range(80)]
        + [{"category": "C", "text": f"c{i}"} for i in range(200)]
    )


@pytest.fixture(scope="module")
def dolly_dataset() -> List[dict]:
    """실제 Dolly-15k 전체(약 15,011개)를 이 파일에서 딱 1번만 로드해 공유.

    Loads the real, full Dolly-15k (~15,011 examples) once for this file
    and shares it across every test that needs it."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return [
        {
            "instruction": row["instruction"],
            "context": row.get("context", ""),
            "response": row["response"],
            "category": row["category"],
        }
        for row in ds
    ]


# ── 더미 데이터: 빠르고 결정적, 네트워크 불필요 ──────────────────────────
# ── Dummy data: fast, deterministic, no network required ──────────────────

def test_min_threshold_resampling():
    dataset = make_dummy_dataset()
    client_data = partition_by_category(dataset, num_clients=8, alpha=0.5, min_category_threshold=150)
    total_b = sum(1 for items in client_data.values() for x in items if x["category"] == "B")
    assert total_b >= 150, "임계값 미만 카테고리가 리샘플링되지 않음"


def test_partition_is_non_iid():
    dataset = make_dummy_dataset()
    client_data = partition_by_category(dataset, num_clients=8, alpha=0.1, min_category_threshold=150)
    score = heterogeneity_score(client_data)
    assert score > 0, "alpha=0.1(강한 non-IID)인데 heterogeneity_score가 0"


def test_alpha_controls_heterogeneity_strength():
    """alpha=0.1 > alpha=0.5 > alpha=1.0 순으로 비IID 강도(heterogeneity_score)가
    커야 함 — Dirichlet 파티셔닝이 의도대로 동작하는지에 대한 일반 검증.

    The non-IID strength (heterogeneity_score) should increase in the order
    alpha=0.1 > alpha=0.5 > alpha=1.0 — a general check that Dirichlet
    partitioning behaves as intended."""
    dataset = make_dummy_dataset()

    scores = {}
    for alpha in [1.0, 0.5, 0.1]:
        client_data = partition_by_category(dataset, num_clients=8, alpha=alpha, min_category_threshold=150, seed=1)
        scores[alpha] = heterogeneity_score(client_data)

    print(f"heterogeneity scores (dummy): {scores}")
    assert scores[0.1] >= scores[0.5] >= scores[1.0] - 0.05, (
        f"alpha가 작을수록 비IID가 강해야 하는데 순서가 어긋남: {scores}"
    )


# ── 실제 Dolly-15k 전체: 네트워크 필요, 실데이터 카테고리 분포로 검증 ──────
# ── Real full Dolly-15k: needs network, verifies against real category distribution ──

@pytest.mark.network
def test_dolly_min_threshold_resampling(dolly_dataset):
    # 실제로 가장 표본이 적은 카테고리를 골라서, 그 원본 수보다 명백히
    # 높은 임계값을 줘야 리샘플링 로직이 실제로 발동한다 (dataset[0]의
    # 카테고리를 그냥 쓰면 이미 임계값을 넘어 있어 아무것도 검증 못 할 수 있음).
    #
    # Pick the category with the fewest raw samples, and set a threshold
    # clearly above its original count — otherwise (e.g. just using
    # dataset[0]'s category) it may already exceed the threshold and the
    # resampling path is never actually exercised.
    raw_counts = Counter(item["category"] for item in dolly_dataset)
    target_category = min(raw_counts, key=raw_counts.get)
    threshold = raw_counts[target_category] + 50

    client_data = partition_by_category(dolly_dataset, num_clients=8, alpha=0.5, min_category_threshold=threshold)
    total_target = sum(1 for items in client_data.values() for x in items if x["category"] == target_category)
    assert total_target >= threshold, f"{target_category} Categories are not resampled below the threshold."


@pytest.mark.network
def test_dolly_partition_is_non_iid(dolly_dataset):
    client_data = partition_by_category(dolly_dataset, num_clients=8, alpha=0.1, min_category_threshold=20)
    summary = summarize_partition(client_data)
    for client_id, counts in summary.items():
        print(f"client {client_id}: {counts}")

    score = heterogeneity_score(client_data)
    assert score > 0, "alpha=0.1 but heterogeneity_score is 0"


# 아래 두 개(0.5, 1.0)는 test_dolly_partition_is_non_iid와 동일한 패턴의
# 육안 확인용 테스트다. score > 0은 사실상 항상 참이라 강한 검증은 아니고,
# "0.1 >= 0.5 >= 1.0" 순서를 실제로 검증하는 건
# test_dolly_alpha_controls_heterogeneity_strength 쪽이다 — 여기 두 테스트는
# 각 alpha에서 partition_by_category가 에러 없이 돌고, 클라이언트별 카테고리
# 분포를 -s 옵션으로 눈으로 볼 수 있게 하는 용도.
#
# The two tests below (0.5, 1.0) follow the same visual-inspection pattern
# as test_dolly_partition_is_non_iid. score > 0 is nearly always true, so
# it isn't a strong check — the real ordering check ("0.1 >= 0.5 >= 1.0")
# lives in test_dolly_alpha_controls_heterogeneity_strength. These two are
# for confirming partition_by_category runs cleanly at each alpha and for
# eyeballing the per-client category distribution with `-s`.

@pytest.mark.network
def test_dolly_partition_is_moderately_non_iid(dolly_dataset):
    client_data = partition_by_category(dolly_dataset, num_clients=8, alpha=0.5, min_category_threshold=20)
    summary = summarize_partition(client_data)
    for client_id, counts in summary.items():
        print(f"client {client_id}: {counts}")

    score = heterogeneity_score(client_data)
    assert score > 0, "alpha=0.5 but heterogeneity_score is 0"


@pytest.mark.network
def test_dolly_partition_is_near_iid(dolly_dataset):
    client_data = partition_by_category(dolly_dataset, num_clients=8, alpha=1.0, min_category_threshold=20)
    summary = summarize_partition(client_data)
    for client_id, counts in summary.items():
        print(f"client {client_id}: {counts}")

    score = heterogeneity_score(client_data)
    assert score > 0, "alpha=1.0 but heterogeneity_score is 0"


@pytest.mark.network
def test_dolly_alpha_controls_heterogeneity_strength(dolly_dataset):
    """실제 Dolly-15k 전체 데이터 기준으로도 alpha 순서대로 비IID 강도가
    커지는지 확인 — 더미 데이터 테스트와 동일한 취지를 실데이터로 재검증.

    Confirms the same alpha ordering on the real full Dolly-15k dataset —
    re-validates the dummy-data test's intent against real data."""
    scores = {}
    for alpha in [1.0, 0.5, 0.1]:
        client_data = partition_by_category(dolly_dataset, num_clients=8, alpha=alpha, min_category_threshold=20, seed=1)
        scores[alpha] = heterogeneity_score(client_data)

    print(f"heterogeneity scores (real Dolly-15k): {scores}")
    assert scores[0.1] >= scores[0.5] >= scores[1.0] - 0.05, (
        f"alpha가 작을수록 비IID가 강해야 하는데 순서가 어긋남: {scores}"
    )
