"""Dirichlet(alpha) non-IID 파티셔닝 + 카테고리 최소 임계값 리샘플링.

Subtask 1.1: 8클라이언트, Dolly-15k를 native task category 기준
Dirichlet(alpha=0.5)로 비IID 분배 (고정값, 스윕 없음).
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np


def partition_by_category(
    dataset: List[dict],
    num_clients: int,
    alpha: float,
    min_category_threshold: int,
    seed: int = 42,
) -> Dict[int, List[dict]]:
    """category 필드를 기준으로 Dirichlet(alpha)로 클라이언트별 비IID 분배.

    alpha가 작을수록(예: 0.1) 강한 비IID, 클수록(예: 1.0) near-IID.
    min_category_threshold 미만인 카테고리는 리샘플링(업샘플링)하여
    각 클라이언트가 최소한의 샘플을 확보하도록 함 (계획서 §5.2, §6 리스크 대응).
    """
    rng = np.random.default_rng(seed)

    by_category = defaultdict(list)
    for item in dataset:
        by_category[item["category"]].append(item)

    for cat, items in by_category.items():
        if len(items) < min_category_threshold:
            deficit = min_category_threshold - len(items)
            resampled = list(rng.choice(items, size=deficit, replace=True))
            by_category[cat] = items + resampled

    client_data = {i: [] for i in range(num_clients)}
    for cat, items in by_category.items():
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(items)).astype(int)[:-1]
        splits = np.split(items, split_points)
        for client_id, split in enumerate(splits):
            client_data[client_id].extend(split.tolist())

    return client_data


def summarize_partition(client_data: Dict[int, List[dict]]) -> Dict[int, Dict[str, int]]:
    """검증용: 클라이언트별 카테고리 분포 요약 (육안 확인, Week 2)."""
    summary = {}
    for client_id, items in client_data.items():
        counts = defaultdict(int)
        for item in items:
            counts[item["category"]] += 1
        summary[client_id] = dict(counts)
    return summary


def heterogeneity_score(client_data: Dict[int, List[dict]]) -> float:
    """파티션이 실제로 얼마나 비IID한지 정량화 (0=완전 균등, 1=완전 편중).
    각 클라이언트의 최다 카테고리 비율의 평균으로 근사.
    Dirichlet(alpha=0.5) 파티셔닝이 실제로 non-IID를 만들어내는지
    검증하는 용도(Week 2 파이프라인 검증).
    """
    ratios = []
    for items in client_data.values():
        if not items:
            continue
        counts = defaultdict(int)
        for item in items:
            counts[item["category"]] += 1
        ratios.append(max(counts.values()) / len(items))
    return sum(ratios) / len(ratios) if ratios else 0.0
