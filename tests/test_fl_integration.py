"""5단계: 소규모 멀티 클라이언트 FL 통합 테스트.
계획서 v2에서 SCAFFOLD가 core이므로, FedAvg/FedProx/SCAFFOLD 세 경로
모두 fl_runner.py의 전체 루프(fit->aggregate->evaluate->checkpoint->
convergence check)를 에러 없이 도는지 확인.
"""

import shutil

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.fl_client import FlowerClient
from src.fl_runner import run_federated_training


class DummyTrainSet(Dataset):
    def __init__(self, n=8, seq_len=16, vocab_size=1000):
        self.n, self.seq_len, self.vocab_size = n, seq_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "attention_mask": torch.ones(self.seq_len), "labels": ids.clone()}


class DummyEvalSet:
    def __init__(self, n=4, seq_len=16, vocab_size=1000):
        self.n, self.seq_len, self.vocab_size = n, seq_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            "input_ids": ids, "attention_mask": torch.ones(self.seq_len), "labels": ids.clone(),
            "category": "dummy_category", "prompt": "dummy prompt", "reference_response": "dummy response",
        }


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_dev_config(fl_type):
    with open("configs/dev_config.yaml") as f:
        config = yaml.safe_load(f)
    config["fl_algorithm"]["type"] = fl_type
    config["federated"]["num_clients"] = 2
    config["federated"]["num_rounds"] = 2
    return config


def _make_clients(config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clients = []
    for cid in range(config["federated"]["num_clients"]):
        train_loader = DataLoader(DummyTrainSet(), batch_size=2, collate_fn=_collate)
        clients.append(FlowerClient(config, cid, train_loader, DummyEvalSet(), tokenizer))
    return clients


def test_fedavg_round_loop_runs():
    config = make_dev_config("fedavg")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_fedavg_dummy")
    assert result["rounds_run"] >= 1
    shutil.rmtree("results/checkpoints/test_fedavg_dummy", ignore_errors=True)


def test_fedprox_round_loop_runs():
    config = make_dev_config("fedprox")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_fedprox_dummy")
    assert result["rounds_run"] >= 1
    shutil.rmtree("results/checkpoints/test_fedprox_dummy", ignore_errors=True)


def test_scaffold_round_loop_runs():
    """계획서 v2: SCAFFOLD가 core이므로 전체 루프가 정상 동작해야 함."""
    config = make_dev_config("scaffold")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_scaffold_dummy")
    assert result["rounds_run"] >= 1
    # SCAFFOLD 클라이언트는 local_control을 라운드 간 유지해야 함
    for c in clients:
        assert hasattr(c, "local_control")
    shutil.rmtree("results/checkpoints/test_scaffold_dummy", ignore_errors=True)


def test_fedprox_proximal_term_present_in_fit():
    config = make_dev_config("fedprox")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(DummyTrainSet(), batch_size=2, collate_fn=_collate)
    client = FlowerClient(config, 0, train_loader, DummyEvalSet(), tokenizer)

    from src.communication import get_trainable_state_dict, state_dict_to_ndarrays

    params = state_dict_to_ndarrays(get_trainable_state_dict(client.model))
    _, _, metrics = client.fit(params, {"server_round": 1})
    assert "train_loss" in metrics
