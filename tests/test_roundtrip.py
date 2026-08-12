"""4단계: 파라미터 왕복(round-trip) 단위 테스트."""

import torch

from src.communication import ndarrays_to_state_dict, state_dict_to_ndarrays


def test_roundtrip_preserves_values():
    original = {"lora_A.weight": torch.randn(8, 16), "lora_B.weight": torch.randn(16, 8)}
    keys = list(original.keys())
    arrays = state_dict_to_ndarrays(original)
    restored = ndarrays_to_state_dict(keys, arrays)
    for k in keys:
        assert torch.allclose(original[k], restored[k], atol=1e-6), f"Mismatch in {k}"


def test_roundtrip_preserves_key_order():
    original = {"a": torch.zeros(2), "b": torch.ones(2), "c": torch.full((2,), 3.0)}
    keys = list(original.keys())
    arrays = state_dict_to_ndarrays(original)
    restored = ndarrays_to_state_dict(keys, arrays)
    assert list(restored.keys()) == keys
