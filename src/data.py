"""Dolly-15k 로딩 + 토크나이징 + train/held-out 스플릿 (계획서 v2 §5.1, §5.3).

흐름:
  1) load_raw_dolly15k()        - HuggingFace datasets로 원본 로딩
  2) build_holdout_split()      - 클라이언트 분배 전 전역 held-out 평가셋 확보
                                   (카테고리별 균등 — §5.4 per-category breakdown 대응)
  3) tokenize_example()         - prompt는 라벨 마스킹(-100), response만 loss 계산
  4) build_client_dataloaders() - partitioning.py 결과 + 토크나이징 -> DataLoader

Loading Dolly-15k + tokenization + train/held-out split (plan v2 §5.1, §5.3).

Flow:
  1) load_raw_dolly15k()        - load the raw data via HuggingFace datasets
  2) build_holdout_split()      - carve out a global held-out eval set before client
                                   distribution (evenly per category — supports the
                                   §5.4 per-category breakdown)
  3) tokenize_example()         - mask the prompt span in labels (-100), compute loss
                                   only on the response
  4) build_client_dataloaders() - combine partitioning.py output with tokenization -> DataLoader
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
    """카테고리별로 균등하게 held-out 평가셋을 뗌 (계획서 §5.3).

    Carve out a held-out eval set evenly across categories (plan §5.3).
    """
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

    Category-stratified subsample for ROUGE-L generation evaluation.

    ROUGE-L requires autoregressive generation, which is far more expensive
    than PPL (a teacher-forcing forward pass). Instead of generating over the
    entire held-out set (thousands of examples) every time, this draws only
    sample_size examples while preserving category proportions, to control
    cost (Subtask 1.2/2.3).
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
    response_ids = tokenizer(item["response"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # 전체 텍스트(prompt+response)를 뒤에서부터 자르면, prompt(특히 closed_qa/
    # summarization처럼 context가 긴 카테고리)만으로도 max_length를 넘는
    # 예제는 response가 통째로 잘려나가 라벨이 전부 -100이 된다. 이런 예제가
    # 한 미니배치(4개)에 몰리면 loss 분모가 0이 되어 NaN이 나고, FedAvg
    # 집계를 거쳐 global_state 전체가 오염된다(실측: alpha=0.1 파일럿에서
    # round 1 val_loss=nan 재현). response 토큰 수만큼은 항상 남기고, 넘치는
    # 만큼만 prompt 앞부분을 자르는 방식으로 바꿔 이 경우를 원천 차단한다 —
    # response_ids는 eos_token을 포함하므로 항상 길이 >= 1이라 라벨이 전부
    # -100이 되는 경우가 생기지 않는다.
    #
    # Truncating the concatenated (prompt+response) text from the end meant
    # that any example whose prompt alone reached max_length (e.g. closed_qa/
    # summarization, which often have long context) lost the response
    # entirely, leaving every label as -100. When enough such examples landed
    # in one micro-batch (4), the loss denominator hit 0 and produced NaN,
    # which corrupted the whole global_state after FedAvg aggregation
    # (reproduced: round 1 val_loss=nan on the alpha=0.1 pilot). This always
    # keeps the response tokens and truncates only the prompt's leading part
    # by however much is needed — response_ids always has length >= 1 (it
    # includes eos_token), so labels can never end up fully -100.
    response_ids = response_ids[:max_length]
    max_prompt_len = max_length - len(response_ids)
    prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)
    attention_mask = [1] * len(input_ids)

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id
        if tokenizer.padding_side == "left":
            input_ids = [pad_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + attention_mask
            labels = [-100] * pad_len + labels
        else:
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

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
    """§5.4 카테고리별 성능 분해용 — category/prompt/reference가 보존된 held-out 데이터셋.

    A held-out dataset for the §5.4 per-category performance breakdown — one
    that preserves category/prompt/reference fields.
    """
    return TokenizedDolly(holdout_eval, tokenizer, max_length)
