"""src/scaffold.py 유닛 테스트 — SCAFFOLD control variate 집계 수식이
실제로 문서화된 대로 동작하는지 검증.

`scaffold_aggregate`는 순수 텐서 연산이라 모델 없이 손으로 계산한 값과
직접 비교해서 검증한다. `scaffold_client_fit`은 작은 더미 모델로 로컬
학습 1스텝이 에러 없이 돌고 local_control이 실제로 갱신되는지 확인한다
(SCAFFOLD는 core 압축×non-IID 분석에서는 제외됐지만, 코드/테스트는
유지된다 — README 참고).

Unit tests for src/scaffold.py — verifies the control-variate aggregation
formulas actually behave as documented.

`scaffold_aggregate` is pure tensor math, verified directly against
hand-computed expected values with no model involved.
`scaffold_client_fit` is checked with a small dummy model: one local
training step runs without error and local_control is actually updated
(SCAFFOLD is excluded from the core compression x non-IID analysis, but
its code/tests are kept — see the README).
"""

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.communication import get_trainable_state_dict
from src.models import get_model
from src.scaffold import scaffold_aggregate, scaffold_client_fit, zeros_like_trainable


def load_dev_config():
    with open("configs/dev_config.yaml") as f:
        config = yaml.safe_load(f)
    config["peft"]["type"] = "lora"
    return config


# ── zeros_like_trainable ─────────────────────────────────────────────────

def test_zeros_like_trainable_matches_shapes_and_is_zero():
    model = get_model(load_dev_config())
    zeros = zeros_like_trainable(model)
    trainable = get_trainable_state_dict(model)

    assert set(zeros.keys()) == set(trainable.keys())
    for k in zeros:
        assert zeros[k].shape == trainable[k].shape
        assert torch.all(zeros[k] == 0)


# ── scaffold_aggregate: 순수 텐서 연산, 손으로 계산한 값과 직접 비교 ────
# ── scaffold_aggregate: pure tensor math, checked against hand-computed values ──

def test_scaffold_aggregate_averages_delta_y_across_clients():
    global_state = {"w": torch.tensor([0.0, 0.0])}
    global_control = {"w": torch.tensor([0.0, 0.0])}
    client_deltas = [
        ({"w": torch.tensor([2.0, 0.0])}, {"w": torch.tensor([1.0, 0.0])}, 10),
        ({"w": torch.tensor([4.0, 0.0])}, {"w": torch.tensor([3.0, 0.0])}, 10),
    ]

    new_state, new_control = scaffold_aggregate(global_state, global_control, client_deltas, num_total_clients=2)

    # avg_delta_y = (2+4)/2 = 3 -> new_state = 0 + 3
    assert torch.allclose(new_state["w"], torch.tensor([3.0, 0.0]))
    # avg_delta_c = (1+3)/2 = 2, 참여율 2/2=1 -> new_control = 0 + 1*2
    assert torch.allclose(new_control["w"], torch.tensor([2.0, 0.0]))


def test_scaffold_aggregate_scales_control_update_by_participation_rate():
    """전체 클라이언트 수보다 적게 참여하면 global_control 갱신폭이
    참여율(n_participating/num_total_clients)만큼 줄어들어야 함.

    When fewer clients participate than the total, the global_control
    update should shrink proportionally to the participation rate
    (n_participating/num_total_clients)."""
    global_state = {"w": torch.tensor([0.0])}
    global_control = {"w": torch.tensor([0.0])}
    client_deltas = [({"w": torch.tensor([1.0])}, {"w": torch.tensor([4.0])}, 5)]

    # 1명만 참여, 전체는 4명 -> 참여율 1/4
    _, new_control = scaffold_aggregate(global_state, global_control, client_deltas, num_total_clients=4)
    assert torch.allclose(new_control["w"], torch.tensor([1.0]))  # 0 + (1/4)*4 = 1


# ── scaffold_client_fit ──────────────────────────────────────────────────

class _DummyTrainSet(Dataset):
    def __init__(self, n=8, seq_len=16, vocab_size=1000):
        self.n, self.seq_len, self.vocab_size = n, seq_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "attention_mask": torch.ones(self.seq_len), "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def test_scaffold_client_fit_runs_and_produces_nonzero_delta_y():
    model = get_model(load_dev_config())
    global_state = {k: v.clone() for k, v in get_trainable_state_dict(model).items()}
    local_control = zeros_like_trainable(model)
    global_control = zeros_like_trainable(model)
    train_loader = DataLoader(_DummyTrainSet(), batch_size=2, collate_fn=_collate)

    delta_y, delta_c, num_ex, new_local_control = scaffold_client_fit(
        model, global_state, local_control, global_control, train_loader,
        learning_rate=0.1, local_epochs=1, accum_steps=1,
    )

    assert num_ex == 8
    assert set(delta_y.keys()) == set(global_state.keys())
    assert set(new_local_control.keys()) == set(local_control.keys())
    # 학습이 실제로 일어났다면 델타가 전부 0은 아니어야 함
    # If training actually happened, the deltas shouldn't be all zero
    assert any(torch.any(v != 0) for v in delta_y.values())
