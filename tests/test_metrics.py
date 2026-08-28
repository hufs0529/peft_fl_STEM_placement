"""src/metrics.py 유닛 테스트 — ROUGE-L 계산, trainable parameter 집계,
VRAM/latency 계측 컨텍스트 매니저, 응답 생성. GPU 없이(CPU) 전부 검증
가능하다(peak_vram_gb는 CPU에서 0.0이 나오는 것 자체가 기대 동작).

Unit tests for src/metrics.py — ROUGE-L computation, trainable parameter
counting, the VRAM/latency measurement context manager, response
generation. All verifiable on CPU with no GPU (peak_vram_gb reading 0.0 on
CPU is itself the expected behaviour).
"""

import time

import pytest
import yaml
from transformers import AutoTokenizer

from src.metrics import compute_rouge_l, count_trainable_parameters, generate_responses, track_vram_and_latency
from src.models import get_model


def load_dev_config():
    with open("configs/dev_config.yaml") as f:
        return yaml.safe_load(f)


# ── compute_rouge_l ──────────────────────────────────────────────────────

def test_compute_rouge_l_identical_strings_scores_near_one():
    preds = ["The quick brown fox jumps over the lazy dog"]
    refs = ["The quick brown fox jumps over the lazy dog"]
    assert compute_rouge_l(preds, refs) > 0.99


def test_compute_rouge_l_disjoint_strings_scores_low():
    preds = ["completely unrelated text here"]
    refs = ["totally different reference sentence"]
    assert compute_rouge_l(preds, refs) < 0.3


def test_compute_rouge_l_averages_across_examples():
    # 첫 예제는 완전 일치, 둘째는 겹치는 단어가 거의 없음 -> 평균은 둘 사이 어딘가
    # First example matches exactly, the second shares almost no words —
    # the average should land strictly between the two extremes.
    preds = ["exact match here", "nothing in common whatsoever"]
    refs = ["exact match here", "zzz yyy xxx www"]
    score = compute_rouge_l(preds, refs)
    assert 0.0 < score < 1.0


def test_compute_rouge_l_empty_input_does_not_crash():
    assert compute_rouge_l([], []) == 0.0


# ── count_trainable_parameters ──────────────────────────────────────────

def test_count_trainable_parameters_lora_is_small_fraction_of_total():
    config = load_dev_config()
    config["peft"]["type"] = "lora"
    model = get_model(config)
    stats = count_trainable_parameters(model)

    assert 0 < stats["trainable_params"] < stats["total_params"]
    assert 0 < stats["trainable_pct"] < 100
    assert stats["trainable_pct"] == pytest.approx(
        100.0 * stats["trainable_params"] / stats["total_params"]
    )


# ── track_vram_and_latency ──────────────────────────────────────────────

def test_track_vram_and_latency_measures_positive_latency():
    with track_vram_and_latency(device="cpu") as stats:
        time.sleep(0.01)
    assert stats["latency_sec"] > 0.0


def test_track_vram_and_latency_zero_vram_on_cpu():
    with track_vram_and_latency(device="cpu") as stats:
        pass
    assert stats["peak_vram_gb"] == 0.0


# ── generate_responses ──────────────────────────────────────────────────

def test_generate_responses_returns_one_string_per_prompt():
    config = load_dev_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config["peft"]["type"] = "lora"
    model = get_model(config)
    model.eval()

    predictions = generate_responses(model, tokenizer, ["Hello", "World"], max_new_tokens=4, device="cpu")
    assert len(predictions) == 2
    assert all(isinstance(p, str) for p in predictions)
