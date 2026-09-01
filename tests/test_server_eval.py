"""src/server_eval.py 유닛 테스트 — 서버가 클라이언트 인터페이스를 거치지
않고 글로벌 모델을 직접 평가하는지 확인 (지도교수 피드백: "evaluation is
based on the test on server, not individual clients").
"""

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.models import get_model
from src.server_eval import evaluate_global_model, evaluate_global_model_generation


def load_dev_config():
    with open("configs/dev_config.yaml") as f:
        config = yaml.safe_load(f)
    config["peft"]["type"] = "lora"
    return config


class _DummyEvalSet(Dataset):
    def __init__(self, n=4, seq_len=16, vocab_size=1000):
        self.n, self.seq_len, self.vocab_size = n, seq_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "attention_mask": torch.ones(self.seq_len), "labels": ids.clone()}


class _DummyGenSet(Dataset):
    def __init__(self, n=2):
        self.n = n
        self.categories = ["open_qa", "closed_qa"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "prompt": f"prompt {idx}",
            "reference_response": f"answer {idx}",
            "category": self.categories[idx % len(self.categories)],
        }


def test_evaluate_global_model_returns_loss_and_count():
    model = get_model(load_dev_config())
    avg_loss, n = evaluate_global_model(model, _DummyEvalSet(n=4), device="cpu")
    assert n == 4
    assert avg_loss > 0


def test_evaluate_global_model_generation_returns_rouge_and_predictions():
    config = load_dev_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_model(config)
    model.eval()

    result = evaluate_global_model_generation(model, tokenizer, _DummyGenSet(n=2), device="cpu")
    assert "rouge_l" in result
    assert len(result["predictions"]) == 2
    assert result["categories"] == ["open_qa", "closed_qa"]
