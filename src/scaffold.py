"""SCAFFOLD 실제 구현 (Karimireddy et al., 2020, Option II 근사).

계획서 v2에서 SCAFFOLD는 stretch가 아니라 core FL 알고리즘입니다(§4.2) —
RQ1("교정 정교함이 양자화와 상호작용하는가")을 답하려면 FedAvg/FedProx/
SCAFFOLD 세 지점이 모두 필요하기 때문입니다.

정확도보다 핵심 메커니즘(control variate로 client drift 보정)이 실제로
작동하는지를 보여주는 데 목적을 둔 간소화 구현입니다 — 이 점을 최종
리포트의 Limitations에 명시할 것 (계획서 §7 Week 6 항목 참고).
"""

from typing import Dict, List, Tuple

import torch

from src.communication import get_trainable_state_dict, set_trainable_state_dict


def zeros_like_trainable(model) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in get_trainable_state_dict(model).items()}


def scaffold_client_fit(
    model,
    global_state_dict: Dict[str, torch.Tensor],
    local_control: Dict[str, torch.Tensor],
    global_control: Dict[str, torch.Tensor],
    train_loader,
    learning_rate: float,
    local_epochs: int,
    accum_steps: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, Dict[str, torch.Tensor]]:
    """한 클라이언트의 SCAFFOLD 로컬 학습 1라운드.
    반환: (delta_y, delta_c, num_examples, new_local_control)
    """
    set_trainable_state_dict(model, global_state_dict)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    param_keys = list(get_trainable_state_dict(model).keys())
    num_examples = 0
    num_local_steps = 0

    model.train()
    for _ in range(local_epochs):
        for step, batch in enumerate(train_loader):
            loss = model(**batch).loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            correction = global_control[name] - local_control[name]
                            param.grad.add_(correction)
                optimizer.step()
                optimizer.zero_grad()
                num_local_steps += 1

            num_examples += batch["input_ids"].shape[0]

    final_state = get_trainable_state_dict(model)

    new_local_control = {}
    delta_c = {}
    denom = max(num_local_steps, 1) * learning_rate
    for k in param_keys:
        c_i_new = local_control[k] - global_control[k] + (global_state_dict[k] - final_state[k]) / denom
        new_local_control[k] = c_i_new
        delta_c[k] = c_i_new - local_control[k]

    delta_y = {k: final_state[k] - global_state_dict[k] for k in param_keys}
    return delta_y, delta_c, num_examples, new_local_control


def scaffold_aggregate(
    global_state_dict: Dict[str, torch.Tensor],
    global_control: Dict[str, torch.Tensor],
    client_deltas: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]],
    num_total_clients: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n_participating = len(client_deltas)
    keys = list(global_state_dict.keys())

    avg_delta_y = {k: torch.zeros_like(global_state_dict[k]) for k in keys}
    avg_delta_c = {k: torch.zeros_like(global_control[k]) for k in keys}

    for delta_y, delta_c, _num_examples in client_deltas:
        for k in keys:
            avg_delta_y[k] += delta_y[k] / n_participating
            avg_delta_c[k] += delta_c[k] / n_participating

    new_global_state = {k: global_state_dict[k] + avg_delta_y[k] for k in keys}
    new_global_control = {
        k: global_control[k] + (n_participating / num_total_clients) * avg_delta_c[k] for k in keys
    }
    return new_global_state, new_global_control
