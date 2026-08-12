"""2단계: 미니어처 모델(gpt2)로 연구계획서의 3개 PEFT 방식(lora/qlora/dora) 배선 검증.
목적: 성능이 아니라 '에러 없이 학습 가능한 파라미터가 잡히는가'.
"""

import yaml
import torch
import pytest

from src.models import get_model
from src.communication import get_trainable_state_dict


def load_dev_config():
    with open("configs/dev_config.yaml") as f:
        return yaml.safe_load(f)


def test_lora_has_trainable_params():
    config = load_dev_config()
    config["peft"]["type"] = "lora"
    model = get_model(config)
    assert len(get_trainable_state_dict(model)) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="qlora(bitsandbytes 4bit)는 GPU 환경에서만 동작")
def test_qlora_has_trainable_params():
    config = load_dev_config()
    config["peft"]["type"] = "qlora"
    model = get_model(config)
    assert len(get_trainable_state_dict(model)) > 0


def test_dora_has_trainable_params():
    """Subtask 2.1 진단 대조군: DoRA (use_dora=True, 양자화 없음)."""
    config = load_dev_config()
    config["peft"]["type"] = "dora"
    model = get_model(config)
    assert len(get_trainable_state_dict(model)) > 0


def test_unknown_peft_type_raises():
    """연구계획서 범위 밖 PEFT type은 ValueError — lora/qlora/dora만 정의됨."""
    config = load_dev_config()
    config["peft"]["type"] = "adapter_tuning"
    try:
        get_model(config)
        assert False, "정의되지 않은 peft type은 ValueError가 발생해야 함"
    except ValueError:
        pass


def test_forward_pass_runs_without_error():
    config = load_dev_config()
    config["peft"]["type"] = "lora"
    model = get_model(config)
    dummy_input = torch.randint(0, 1000, (1, 8))
    output = model(input_ids=dummy_input, labels=dummy_input)
    assert output.loss is not None
