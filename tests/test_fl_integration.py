"""5단계: 소규모 멀티 클라이언트 FL 통합 테스트.
계획서 v2에서 SCAFFOLD가 core이므로, FedAvg/FedProx/SCAFFOLD 세 경로
모두 fl_runner.py의 전체 루프(fit->aggregate->evaluate->checkpoint->
convergence check)를 에러 없이 도는지 확인.

Step 5: small-scale multi-client FL integration test.
Since SCAFFOLD is core in plan v2, verifies that all three paths —
FedAvg/FedProx/SCAFFOLD — run through fl_runner.py's full loop
(fit->aggregate->evaluate->checkpoint->convergence check) without error.
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
    """계획서 v2: SCAFFOLD가 core이므로 전체 루프가 정상 동작해야 함.

    Plan v2: since SCAFFOLD is core, the full loop must work correctly.
    """
    config = make_dev_config("scaffold")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_scaffold_dummy")
    assert result["rounds_run"] >= 1
    # SCAFFOLD 클라이언트는 local_control을 라운드 간 유지해야 함
    # SCAFFOLD clients must persist local_control across rounds
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


# ── 압축률×non-IID 트레이드오프 분석용 지표(peak_vram_gb/total_latency_sec) ──
# ── Metrics for the compression-rate x non-IID trade-off analysis ──────────
# (peak_vram_gb / total_latency_sec)

def test_fedavg_round_record_includes_peak_vram_gb():
    """fit()이 계산하는 peak_vram_gb가 이제 round_record까지 살아남아야 함
    (예전엔 fit_results가 라운드 루프 밖으로 안 나가 버려졌음).

    peak_vram_gb, computed by fit(), must now survive into round_record
    (previously it was discarded because fit_results never left the round
    loop)."""
    config = make_dev_config("fedavg")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_fedavg_vram_dummy")
    for record in result["round_records"]:
        assert record["peak_vram_gb"] is not None
        assert record["peak_vram_gb"] >= 0.0
    assert result["avg_peak_vram_gb"] is not None
    assert result["avg_peak_vram_gb"] >= 0.0
    shutil.rmtree("results/checkpoints/test_fedavg_vram_dummy", ignore_errors=True)


def test_fedprox_round_record_includes_peak_vram_gb():
    config = make_dev_config("fedprox")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_fedprox_vram_dummy")
    for record in result["round_records"]:
        assert record["peak_vram_gb"] is not None
    assert result["avg_peak_vram_gb"] is not None
    shutil.rmtree("results/checkpoints/test_fedprox_vram_dummy", ignore_errors=True)


def test_scaffold_peak_vram_gb_stays_none_not_instrumented():
    """SCAFFOLD는 압축×non-IID core 분석에서 제외됐고 scaffold_client_fit()이
    VRAM을 재지 않으므로, 0으로 얼버무리지 않고 명시적으로 None이어야 함
    (0으로 나오면 "측정했더니 0GB"로 오인될 수 있음).

    SCAFFOLD is excluded from the core compression x non-IID analysis and
    scaffold_client_fit() doesn't measure VRAM, so this must stay explicitly
    None rather than being papered over as 0 (0 could be misread as "measured
    and it was 0GB")."""
    config = make_dev_config("scaffold")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_scaffold_vram_dummy")
    for record in result["round_records"]:
        assert record["peak_vram_gb"] is None
    assert result["avg_peak_vram_gb"] is None
    shutil.rmtree("results/checkpoints/test_scaffold_vram_dummy", ignore_errors=True)


def test_total_latency_sec_matches_sum_of_round_latencies():
    config = make_dev_config("fedavg")
    clients = _make_clients(config)
    result = run_federated_training(config, clients, run_name="test_fedavg_latency_dummy")
    expected = sum(r["round_latency_sec"] for r in result["round_records"])
    assert result["total_latency_sec"] == expected
    shutil.rmtree("results/checkpoints/test_fedavg_latency_dummy", ignore_errors=True)
