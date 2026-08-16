"""
Stage 2: verifies the wiring of the research proposal's 3 PEFT methods
(lora/qlora/dora) using a miniature model (gpt2).
Goal: not performance, but whether trainable parameters are captured
without error.
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
    """
    Subtask 2.1 diagnostic control: DoRA (use_dora=True, no quantization).
    """
    config = load_dev_config()
    config["peft"]["type"] = "dora"
    model = get_model(config)
    assert len(get_trainable_state_dict(model)) > 0


def test_unknown_peft_type_raises():
    """
    A PEFT type outside the scope of the research proposal raises
    ValueError — only lora/qlora/dora are defined.
    """
    config = load_dev_config()
    config["peft"]["type"] = "adapter_tuning"
    try:
        get_model(config)
        assert False, "Expected ValueError for unknown PEFT type"
    except ValueError:
        pass


def test_forward_pass_runs_without_error():
    config = load_dev_config()
    config["peft"]["type"] = "lora"
    model = get_model(config)
    dummy_input = torch.randint(0, 1000, (1, 8))
    output = model(input_ids=dummy_input, labels=dummy_input)
    assert output.loss is not None
