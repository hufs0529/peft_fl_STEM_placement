"""수동 FL 라운드 루프 — 연구계획서의 핵심 실행 엔진.

flwr.simulation.start_simulation()은 고정 num_rounds만큼 무조건 실행되고
라운드 중간에 수렴 기준으로 멈추거나 체크포인트에서 재개하기 어려워,
Strategy 집계 로직을 직접 구현하는 수동 루프를 씁니다. 배선되는 것:

  - 수렴 기준(Subtask 1.2)으로 조기 종료
  - 매 라운드 체크포인트 저장 + 재시작 시 자동 재개
  - Subtask 2.3 추가 분석(카테고리별 분해, 총 통신비용)
  - FedAvg/FedProx는 weighted-average 집계, SCAFFOLD는 scaffold.py 경로
  - (선택) W&B 실시간 로깅

평가 설계: PPL은 매 라운드, 대표 클라이언트(clients[0]) 1개로만 계산한다.
집계 직후 모든 클라이언트는 동일한 global 파라미터로 덮어써지고 동일한
공용 held-out set을 보므로, 8명을 전부 평가해도 결과가 (부동소수점 오차
제외) 동일하다 — 나머지 7번은 순수 중복 계산이라 제거했다. 이 설계상
클라이언트별 fairness variance는 의미가 없어(모두 동일값) 계산하지 않는다
(README 알려진 한계 참고). ROUGE-L(생성 필요, 훨씬 비쌈)은 매 라운드가
아니라 루프 종료 후 최종 global 파라미터 기준 1회만, 카테고리 층화
샘플(rouge_eval_dataset)에 대해서만 계산한다.
"""

import json
import os
import time
from typing import List

import torch

from src.checkpointing import resume_or_start_fresh, save_checkpoint
from src.communication import compute_payload_bytes, get_trainable_state_dict, set_trainable_state_dict
from src.convergence import ConvergenceTracker
from src.scaffold import scaffold_aggregate, scaffold_client_fit, zeros_like_trainable


def _init_wandb(config: dict, run_name: str):
    if not config.get("logging", {}).get("use_wandb", False):
        return None
    import wandb

    return wandb.init(
        project=config["logging"].get("wandb_project", "dolly15k-fl-peft"),
        name=run_name, config=config, reinit=True,
    )


def run_federated_training(
    config: dict,
    clients: List,
    run_name: str,
    checkpoint_base_dir: str = "results/checkpoints",
    log_dir: str = "results/logs",
) -> dict:
    fl_type = config["fl_algorithm"]["type"]
    num_rounds = config["federated"]["num_rounds"]
    fraction_fit = config["federated"].get("fraction_fit", 1.0)
    num_clients = len(clients)

    os.makedirs(log_dir, exist_ok=True)
    round_log_path = os.path.join(log_dir, f"{run_name}_rounds.jsonl")
    wb = _init_wandb(config, run_name)

    start_round, resumed_state, extra = resume_or_start_fresh(checkpoint_base_dir, run_name)
    template_state = get_trainable_state_dict(clients[0].model)
    global_state = resumed_state if resumed_state is not None else {k: v.clone() for k, v in template_state.items()}
    global_control = extra.get("global_control") or zeros_like_trainable(clients[0].model)
    if fl_type == "scaffold":
        for c in clients:
            if not hasattr(c, "local_control"):
                c.local_control = zeros_like_trainable(c.model)

    convergence = ConvergenceTracker(
        min_improvement_pct=config.get("convergence", {}).get("min_improvement_pct", 1.0),
        patience_rounds=config.get("convergence", {}).get("patience_rounds", 3),
    )
    convergence.history = extra.get("val_loss_history", [])

    payload_bytes_per_round = compute_payload_bytes(global_state)
    round_records = []
    converged_round = None

    for round_num in range(start_round + 1, num_rounds + 1):
        round_start = time.perf_counter()
        num_selected = max(1, int(num_clients * fraction_fit))
        selected_idx = torch.randperm(num_clients)[:num_selected].tolist()

        fit_results = []
        for idx in selected_idx:
            client = clients[idx]
            if fl_type == "scaffold":
                delta_y, delta_c, num_ex, new_local_control = scaffold_client_fit(
                    client.model, global_state, client.local_control, global_control,
                    client.train_loader, config["training"]["learning_rate"],
                    config["federated"].get("local_epochs", 1), config["training"]["grad_accumulation_steps"],
                )
                client.local_control = new_local_control
                fit_results.append({"delta_y": delta_y, "delta_c": delta_c, "num_examples": num_ex})
            else:
                set_trainable_state_dict(client.model, global_state)
                params, num_ex, metrics = client.fit([v.numpy() for v in global_state.values()], {"server_round": round_num})
                fit_results.append({"params": params, "num_examples": num_ex, "metrics": metrics})

        if fl_type == "scaffold":
            client_deltas = [(r["delta_y"], r["delta_c"], r["num_examples"]) for r in fit_results]
            global_state, global_control = scaffold_aggregate(global_state, global_control, client_deltas, num_clients)
        else:
            # FedAvg/FedProx 공통 weighted-average 집계.
            # FedProx의 proximal term은 fl_client.py의 fit()에서 이미 loss에 반영됨.
            total_ex = sum(r["num_examples"] for r in fit_results)
            keys = list(global_state.keys())
            new_state = {k: torch.zeros_like(global_state[k]) for k in keys}
            for r in fit_results:
                weight = r["num_examples"] / total_ex
                for i, k in enumerate(keys):
                    new_state[k] += weight * torch.tensor(r["params"][i])
            global_state = new_state

        # PPL 전용 평가: 대표 클라이언트 1개만 (위 모듈 docstring 참고 — 8명 다
        # 평가해도 결과가 동일해 나머지는 순수 중복 계산이었음).
        set_trainable_state_dict(clients[0].model, global_state)
        val_loss, _n, _metrics = clients[0].evaluate(
            [v.numpy() for v in global_state.values()], {"run_generation_metrics": False}
        )

        round_record = {
            "round": round_num,
            "val_loss": val_loss,
            "val_perplexity": float(torch.exp(torch.tensor(val_loss))),
            "round_latency_sec": time.perf_counter() - round_start,
            "communication_bytes_this_round": payload_bytes_per_round * len(selected_idx),
        }
        round_records.append(round_record)
        with open(round_log_path, "a") as f:
            f.write(json.dumps(round_record) + "\n")
        if wb is not None:
            import wandb
            wandb.log(round_record, step=round_num)

        print(f"[{run_name}] round {round_num}/{num_rounds} — val_loss={val_loss:.4f} "
              f"latency={round_record['round_latency_sec']:.1f}s")

        save_checkpoint(
            checkpoint_base_dir, run_name, round_num, global_state,
            extra_state={"global_control": global_control, "val_loss_history": convergence.history + [val_loss]},
        )

        has_converged = convergence.update(val_loss)
        if has_converged:
            converged_round = round_num
            print(f"[{run_name}] 수렴 기준 충족 — round {round_num}에서 조기 종료")
            break

    # 루프 종료(수렴 조기종료 또는 num_rounds 도달) 후, 최종 global 파라미터
    # 기준으로 ROUGE-L 생성 평가를 정확히 1회만 수행 (Subtask 1.2 task
    # performance 지표 중 생성이 필요한 부분 — 비용 통제를 위해 라운드마다
    # 돌리지 않는다).
    set_trainable_state_dict(clients[0].model, global_state)
    _, _, final_metrics = clients[0].evaluate(
        [v.numpy() for v in global_state.values()], {"run_generation_metrics": True}
    )
    if "rouge_l" in final_metrics:
        round_records[-1]["rouge_l"] = final_metrics["rouge_l"]
        with open(round_log_path) as f:
            lines = f.readlines()
        lines[-1] = json.dumps(round_records[-1]) + "\n"
        with open(round_log_path, "w") as f:
            f.writelines(lines)
        if wb is not None:
            import wandb
            wandb.log({"rouge_l": final_metrics["rouge_l"]}, step=round_records[-1]["round"])

        gen_path = os.path.join(log_dir, f"{run_name}_generations.jsonl")
        predictions = final_metrics.get("_generated_predictions", [])
        categories = final_metrics.get("_per_example_categories", [])
        if predictions:
            with open(gen_path, "w") as gf:
                for i, pred in enumerate(predictions):
                    gf.write(json.dumps({
                        "instruction": clients[0].rouge_eval_dataset[i]["prompt"],
                        "reference": clients[0].rouge_eval_dataset[i]["reference_response"],
                        "prediction": pred,
                        "category": categories[i] if i < len(categories) else None,
                    }) + "\n")

    total_comm_bytes = sum(r["communication_bytes_this_round"] for r in round_records)
    if wb is not None:
        import wandb
        wandb.finish()

    return {
        "run_name": run_name,
        "rounds_run": len(round_records),
        "converged_round": converged_round,
        "final_val_loss": round_records[-1]["val_loss"],
        "val_perplexity": round_records[-1]["val_perplexity"],
        "rouge_l": round_records[-1].get("rouge_l"),
        "total_communication_bytes": total_comm_bytes,
        "round_records": round_records,
    }
