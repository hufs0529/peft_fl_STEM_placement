"""Dolly-15k 데이터 파이프라인 유닛 테스트 — 프롬프트 포맷, 라벨 마스킹,
held-out 분할, 카테고리 층화 샘플링, DataLoader 구성. 실제 Dolly-15k
다운로드가 필요한 load_raw_dolly15k()만 @pytest.mark.network로 분리한다
(기본 실행에서 빼려면 `pytest tests/ -m "not network"`).

Unit tests for the Dolly-15k data pipeline — prompt formatting, label
masking, the held-out split, category-stratified sampling, DataLoader
construction. Only load_raw_dolly15k() (which needs the real Dolly-15k
download) is marked @pytest.mark.network (exclude it via
`pytest tests/ -m "not network"`).
"""

from collections import Counter

import pytest
import yaml
from transformers import AutoTokenizer

from src.data import (
    build_category_tagged_holdout,
    build_client_dataloaders,
    build_holdout_split,
    format_prompt,
    load_raw_dolly15k,
    sample_for_generation_eval,
    tokenize_example,
)


def make_dummy_raw(n_per_category: int):
    categories = ["open_qa", "closed_qa", "creative_writing"]
    items = []
    for cat in categories:
        for i in range(n_per_category):
            items.append({
                "instruction": f"{cat} instruction {i}",
                "context": "",
                "response": f"{cat} response {i}",
                "category": cat,
            })
    return items


@pytest.fixture(scope="module")
def tokenizer():
    with open("configs/dev_config.yaml") as f:
        config = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── format_prompt ───────────────────────────────────────────────────────

def test_format_prompt_without_context_has_no_context_block():
    item = {"instruction": "Say hi", "context": "", "response": "Hi!"}
    prompt = format_prompt(item)
    assert "### Instruction:\nSay hi" in prompt
    assert "### Context:" not in prompt
    assert prompt.endswith("### Response:\n")


def test_format_prompt_with_context_includes_context_block():
    item = {"instruction": "Summarize", "context": "Some passage.", "response": "..."}
    prompt = format_prompt(item)
    assert "### Context:\nSome passage." in prompt


# ── tokenize_example: 라벨 마스킹이 핵심 정합성 지점 ────────────────────
# ── tokenize_example: label masking is the critical correctness point ──

def test_tokenize_example_padding_positions_are_masked(tokenizer):
    item = {"instruction": "Say hi", "context": "", "response": "Hi!", "category": "open_qa"}
    out = tokenize_example(item, tokenizer, max_length=64)

    labels = out["labels"].tolist()
    attention_mask = out["attention_mask"].tolist()
    for lab, mask in zip(labels, attention_mask):
        if mask == 0:
            assert lab == -100


def test_tokenize_example_prompt_span_masked_response_span_not(tokenizer):
    """모듈 docstring이 명시한 핵심 계약: prompt 구간은 전부 -100,
    response의 첫 토큰부터는 실제 input_ids와 동일(마스킹 안 됨).
    padding_side가 left/right 어느 쪽이든 이 성질이 성립해야 한다.

    The core contract stated in the module docstring: the prompt span is
    entirely -100, and from the response's first token onward, labels
    match input_ids exactly (not masked). Must hold regardless of
    padding_side."""
    item = {"instruction": "Say hi", "context": "", "response": "Hi there!", "category": "open_qa"}
    prompt = format_prompt(item)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    out = tokenize_example(item, tokenizer, max_length=64)
    input_ids = out["input_ids"].tolist()
    labels = out["labels"].tolist()
    attention_mask = out["attention_mask"].tolist()

    content_start = attention_mask.index(1) if 1 in attention_mask else 0
    prompt_len = min(len(prompt_ids), 64)

    for i in range(content_start, content_start + prompt_len):
        assert labels[i] == -100, f"prompt 위치 {i}가 마스킹되지 않음"

    resp_start = content_start + prompt_len
    if resp_start < len(labels) and attention_mask[resp_start] == 1:
        assert labels[resp_start] == input_ids[resp_start], "response 첫 토큰이 마스킹돼 있음"


def test_tokenize_example_labels_and_input_ids_same_shape(tokenizer):
    item = {"instruction": "x", "context": "", "response": "y", "category": "open_qa"}
    out = tokenize_example(item, tokenizer, max_length=32)
    assert out["labels"].shape == out["input_ids"].shape == out["attention_mask"].shape


def test_tokenize_example_preserves_category_and_reference(tokenizer):
    item = {"instruction": "x", "context": "", "response": "the answer", "category": "creative_writing"}
    out = tokenize_example(item, tokenizer, max_length=32)
    assert out["category"] == "creative_writing"
    assert out["reference_response"] == "the answer"


# ── build_holdout_split ─────────────────────────────────────────────────

def test_build_holdout_split_is_even_per_category():
    raw = make_dummy_raw(n_per_category=10)
    train_pool, holdout_eval = build_holdout_split(raw, holdout_fraction=0.2)

    holdout_counts = Counter(item["category"] for item in holdout_eval)
    train_counts = Counter(item["category"] for item in train_pool)

    for cat in ("open_qa", "closed_qa", "creative_writing"):
        assert holdout_counts[cat] == 2
        assert train_counts[cat] == 8


def test_build_holdout_split_no_overlap_between_train_and_holdout():
    raw = make_dummy_raw(n_per_category=10)
    train_pool, holdout_eval = build_holdout_split(raw, holdout_fraction=0.2)
    train_texts = {item["instruction"] for item in train_pool}
    holdout_texts = {item["instruction"] for item in holdout_eval}
    assert train_texts.isdisjoint(holdout_texts)


# ── sample_for_generation_eval ──────────────────────────────────────────

def test_sample_for_generation_eval_respects_category_stratification():
    raw = make_dummy_raw(n_per_category=20)
    sample = sample_for_generation_eval(raw, sample_size=9)
    counts = Counter(item["category"] for item in sample)
    assert len(sample) == 9
    assert all(c == 3 for c in counts.values())


# ── build_client_dataloaders / build_category_tagged_holdout ──────────

def test_build_client_dataloaders_produces_one_loader_pair_per_client(tokenizer):
    raw = make_dummy_raw(2)
    client_data = {0: raw[:3], 1: raw[3:6]}
    holdout_eval = raw[:2]
    loaders = build_client_dataloaders(client_data, holdout_eval, tokenizer, max_length=32, micro_batch_size=2)

    assert set(loaders.keys()) == {0, 1}
    for cid in loaders:
        assert "train" in loaders[cid]
        assert "eval" in loaders[cid]


def test_build_category_tagged_holdout_preserves_category(tokenizer):
    holdout_eval = make_dummy_raw(1)[:2]
    ds = build_category_tagged_holdout(holdout_eval, tokenizer, max_length=32)
    assert len(ds) == 2
    assert ds[0]["category"] in {"open_qa", "closed_qa", "creative_writing"}


# ── 실제 Dolly-15k 다운로드가 필요한 부분만 네트워크 마커로 분리 ────────
# ── Only the part that needs the real Dolly-15k download is network-marked ──

@pytest.mark.network
def test_load_raw_dolly15k_has_expected_shape_and_fields():
    raw = load_raw_dolly15k()
    assert len(raw) > 15000
    assert set(raw[0].keys()) == {"instruction", "context", "response", "category"}
