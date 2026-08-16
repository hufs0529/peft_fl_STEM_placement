from collections import OrderedDict
from typing import Dict, List
import torch


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract CPU-detached state dict containing only trainable parameters (e.g., PEFT/LoRA adapters)."""
    return {
        k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad
    }


def set_trainable_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Update model's trainable parameters in-place from a state dictionary."""
    model_state = dict(model.named_parameters())
    for k, v in state_dict.items():
        # Copy values into the parameter tensor while maintaining the target device
        model_state[k].data.copy_(v.to(model_state[k].device))


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List:
    """Convert PyTorch state dictionary tensors to a list of NumPy ndarrays (for Flower serialization)."""
    return [v.numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(
    keys: List[str], arrays: List
) -> Dict[str, torch.Tensor]:
    """Reconstruct a PyTorch state dictionary from param keys and NumPy ndarrays."""
    return OrderedDict((k, torch.tensor(a)) for k, a in zip(keys, arrays))


def compute_payload_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Calculate the total payload size in bytes for the given state dict per round."""
    return sum(v.numel() * v.element_size() for v in state_dict.values())