"""Subtask 1.2 core metrics: Trainable Parameters, Task Performance(PPL/ROUGE-L),
Peak VRAM, Training Latency, Convergence Speed(round_records로부터 계산), Communication Cost.

Subtask 1.2 core metrics: Trainable Parameters, Task Performance (PPL/ROUGE-L),
Peak VRAM, Training Latency, Convergence Speed (computed from round_records),
Communication Cost.
"""

import time
from contextlib import contextmanager
from typing import Dict, List

import torch


@contextmanager
def track_vram_and_latency(device: str = "cuda"):
    """with 블록 종료 시 {'peak_vram_gb', 'latency_sec'} 채움.

    Fills {'peak_vram_gb', 'latency_sec'} when the with block exits.
    """
    stats: Dict[str, float] = {}
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["latency_sec"] = time.perf_counter() - start
        if device == "cuda" and torch.cuda.is_available():
            stats["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            stats["peak_vram_gb"] = 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def generate_responses(model, tokenizer, prompts: List[str], max_new_tokens: int = 128, device: str = "cuda") -> List[str]:
    predictions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return predictions


def count_trainable_parameters(model) -> Dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }
