"""계획서 v2 §6 리스크 대응책 검증: 체크포인트 저장 후 재개가 실제로 되는가."""

import shutil

import torch

from src.checkpointing import resume_or_start_fresh, save_checkpoint

TEST_CHECKPOINT_DIR = "results/checkpoints/_test_ckpt"


def teardown_module(module):
    shutil.rmtree(TEST_CHECKPOINT_DIR, ignore_errors=True)


def test_save_and_resume_round_trip():
    state = {"lora_A.weight": torch.randn(4, 8)}
    save_checkpoint(TEST_CHECKPOINT_DIR, "run1", round_num=1, global_state_dict=state, extra_state={"foo": 1})
    save_checkpoint(TEST_CHECKPOINT_DIR, "run1", round_num=2, global_state_dict=state, extra_state={"foo": 2})

    start_round, resumed_state, extra = resume_or_start_fresh(TEST_CHECKPOINT_DIR, "run1")
    assert start_round == 2, "가장 최신 라운드(2)에서 재개해야 함"
    assert extra["foo"] == 2
    assert torch.allclose(resumed_state["lora_A.weight"], state["lora_A.weight"])


def test_fresh_start_when_no_checkpoint():
    start_round, resumed_state, extra = resume_or_start_fresh(TEST_CHECKPOINT_DIR, "nonexistent_run")
    assert start_round == 0
    assert resumed_state is None
    assert extra == {}
