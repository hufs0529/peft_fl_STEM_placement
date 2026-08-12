"""Dolly-15k 로딩 + 토크나이징 + train/held-out 스플릿 (계획서 v2 §5.1, §5.3).

흐름:
  1) load_raw_dolly15k()        - HuggingFace datasets로 원본 로딩
  2) build_holdout_split()      - 클라이언트 분배 전 전역 held-out 평가셋 확보
                                   (카테고리별 균등 — §5.4 per-category breakdown 대응)
  3) tokenize_example()         - prompt는 라벨 마스킹(-100), response만 loss 계산
  4) build_client_dataloaders() - partitioning.py 결과 + 토크나이징 -> DataLoader
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "{context_block}"
    "### Response:\n"
)


def load_raw_dolly15k() -> List[dict]:
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


def build_holdout_split(
    dataset: List[dict], holdout_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """카테고리별로 균등하게 held-out 평가셋을 뗌 (계획서 §5.3)."""
    import random

    rng = random.Random(seed)
    by_category = defaultdict(list)
    for item in dataset:
        by_category[item["category"]].append(item)

    train_pool, holdout_eval = [], []
    for category, items in by_category.items():
        rng.shuffle(items)
        n_holdout = max(1, int(len(items) * holdout_fraction))
        holdout_eval.extend(items[:n_holdout])
        train_pool.extend(items[n_holdout:])

    return train_pool, holdout_eval


def sample_for_generation_eval(holdout_eval: List[dict], sample_size: int, seed: int = 42) -> List[dict]:
    """ROUGE-L 생성 평가용 카테고리 층화 서브샘플.

    ROUGE-L은 자기회귀 생성이 필요해 PPL(teacher-forcing 순전파)보다 훨씬
    비싸다. held-out 전체(수천 개)를 매번 생성하는 대신, 카테고리 비율을
    유지한 채 sample_size개만 뽑아 비용을 통제한다 (Subtask 1.2/2.3).
    """
    import random

    rng = random.Random(seed)
    by_category = defaultdict(list)
    for item in holdout_eval:
        by_category[item["category"]].append(item)

    n_categories = max(len(by_category), 1)
    per_category = max(1, sample_size // n_categories)

    sampled = []
    for items in by_category.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:per_category])

    rng.shuffle(sampled)
    return sampled[:sample_size]


def format_prompt(item: dict) -> str:
    context_block = f"### Context:\n{item['context']}\n\n" if item.get("context") else ""
    return PROMPT_TEMPLATE.format(instruction=item["instruction"], context_block=context_block)


def tokenize_example(item: dict, tokenizer, max_length: int) -> dict:
    prompt = format_prompt(item)
    full_text = prompt + item["response"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=False
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)

    prompt_len = min(len(prompt_ids), max_length)
    for i in range(prompt_len):
        labels[i] = -100
    for i, mask in enumerate(attention_mask):
        if mask == 0:
            labels[i] = -100

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "category": item["category"],
        "reference_response": item["response"],
        "prompt": prompt,
    }


class TokenizedDolly(Dataset):
    def __init__(self, raw_items: List[dict], tokenizer, max_length: int):
        self.items = raw_items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return tokenize_example(self.items[idx], self.tokenizer, self.max_length)


def _collate_for_training(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def build_client_dataloaders(
    client_data: Dict[int, List[dict]],
    holdout_eval: List[dict],
    tokenizer,
    max_length: int,
    micro_batch_size: int,
) -> Dict[int, dict]:
    holdout_ds = TokenizedDolly(holdout_eval, tokenizer, max_length)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=micro_batch_size, shuffle=False, collate_fn=_collate_for_training
    )

    result = {}
    for client_id, items in client_data.items():
        train_ds = TokenizedDolly(items, tokenizer, max_length)
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch_size, shuffle=True, collate_fn=_collate_for_training
        )
        result[client_id] = {"train": train_loader, "eval": holdout_loader}

    return result


def build_category_tagged_holdout(holdout_eval: List[dict], tokenizer, max_length: int) -> Dataset:
    """§5.4 카테고리별 성능 분해용 — category/prompt/reference가 보존된 held-out 데이터셋."""
    return TokenizedDolly(holdout_eval, tokenizer, max_length)
