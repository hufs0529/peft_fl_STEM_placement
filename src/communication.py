"""FL 통신용 파라미터 추출/왕복(round-trip) 로직.

Parameter extraction/round-trip logic for FL communication.
"""

from collections import OrderedDict
from typing import Dict, List

import torch


def get_trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def set_trainable_state_dict(model, state_dict: Dict[str, torch.Tensor]):
    model_state = dict(model.named_parameters())
    for k, v in state_dict.items():
        model_state[k].data.copy_(v.to(model_state[k].device))


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List:
    return [v.numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(keys: List[str], arrays: List) -> Dict[str, torch.Tensor]:
    return OrderedDict((k, torch.tensor(a)) for k, a in zip(keys, arrays))


def compute_payload_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """§5.4 총 통신비용(payload x R) 재료 — 라운드당 payload 크기 측정.

    Ingredient for §5.4 total communication cost (payload x R) — measures
    the payload size per round.
    """
    return sum(v.numel() * v.element_size() for v in state_dict.values())
