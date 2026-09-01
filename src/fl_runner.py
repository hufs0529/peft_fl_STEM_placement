"""수동 FL 라운드 루프 — 연구계획서의 핵심 실행 엔진.

flwr.simulation.start_simulation()은 고정 num_rounds만큼 무조건 실행되고
라운드 중간에 수렴 기준으로 멈추거나 체크포인트에서 재개하기 어려워,
Strategy 집계 로직을 직접 구현하는 수동 루프를 씁니다. 배선되는 것:

  - 수렴 기준(Subtask 1.2)으로 조기 종료
  - 매 라운드 체크포인트 저장 + 재시작 시 자동 재개
  - Subtask 2.3 추가 분석(카테고리별 분해, 총 통신비용)
  - FedAvg/FedProx는 weighted-average 집계, SCAFFOLD는 scaffold.py 경로
  - (선택) W&B 실시간 로깅

압축률×non-IID 트레이드오프 분석(지도교수 피드백)을 위해, FedAvg/FedProx
클라이언트의 fit()이 이미 계산하던 peak_vram_gb/latency_sec(src/metrics.py)를
그동안 round_record에 집계하지 않고 버리고 있었다 — 이번에 round_record에
peak_vram_gb(참여 클라이언트 평균)를, 최종 반환값에 avg_peak_vram_gb/
total_latency_sec를 추가해 압축률×α 상관분석에 쓸 수 있게 했다. SCAFFOLD는
scaffold_client_fit()이 이 계측을 하지 않아 peak_vram_gb가 None으로
남는다 — core 분석(압축×non-IID)에서는 SCAFFOLD가 애초에 빠지므로
의도적으로 계측을 추가하지 않았다.

평가 설계 (지도교수 피드백: "evaluation is based on the test on server,
not individual clients" 반영): PPL은 매 라운드 서버가 src/server_eval.py를
통해 직접 계산한다 — Flower의 NumPyClient.evaluate() 인터페이스(클라이언트
쪽 메서드)를 거치지 않는다. 평가엔 어떤 클라이언트 객체에도 속하지 않는
**서버 전용 모델 인스턴스(server_model)**를 쓴다 — 8개 클라이언트가 각자
자기 모델을 GPU에 들고 있는 것과 별개로, 서버도 자기 모델을 하나 더
들고 있는 구조다(둘 다 같은 프로세스/GPU를 쓰는 로컬 시뮬레이션이라
실제로는 모델 인스턴스가 9개 공존한다). 평가 시점엔 그 라운드 fit()
결과를 집계(aggregate)한 global_state가 server_model에 로드돼 있다.
집계 직후 모든 클라이언트는 동일한 global 파라미터로 덮어써지고 동일한
공용 held-out set을 보므로, 몇 번을 평가하든 결과가 (부동소수점 오차
제외) 동일하다 — 그래서 서버는 이 held-out을 딱 1번만 평가한다. 이
설계상 클라이언트별 fairness variance는 의미가 없어(모두 동일값)
계산하지 않는다(README 알려진 한계 참고). ROUGE-L(생성 필요, 훨씬
비쌈)도 매 라운드가 아니라 루프 종료 후 최종 global_state 기준 1회만,
카테고리 층화 샘플(rouge_eval_dataset)에 대해 서버가 직접 계산한다.

Manual FL round loop — the core execution engine of the research plan.

flwr.simulation.start_simulation() unconditionally runs for a fixed
num_rounds and makes it hard to stop mid-loop on a convergence criterion or
resume from a checkpoint, so we use a manual loop that implements the
Strategy aggregation logic directly. What is wired in:

  - Early stopping via the convergence criterion (Subtask 1.2)
  - Checkpoint save every round + automatic resume on restart
  - Subtask 2.3 additional analysis (per-category breakdown, total
    communication cost)
  - FedAvg/FedProx use weighted-average aggregation, SCAFFOLD uses the
    scaffold.py path
  - (Optional) real-time W&B logging

For the compression-rate × non-IID trade-off analysis (advisor feedback):
FedAvg/FedProx clients' fit() already computed peak_vram_gb/latency_sec
(src/metrics.py), but round_record was discarding them instead of logging
them. This adds peak_vram_gb (averaged over participating clients) to
round_record, and avg_peak_vram_gb/total_latency_sec to the final return
value, so they can be used in the compression × α correlation analysis.
SCAFFOLD's scaffold_client_fit() does not perform this measurement, so its
peak_vram_gb stays None — intentionally not instrumented, since SCAFFOLD is
already excluded from the core (compression × non-IID) analysis.

Evaluation design (reflects advisor feedback: "evaluation is based on the
test on server, not individual clients"): PPL is computed every round by
the server itself, directly via src/server_eval.py — it does not go
through Flower's NumPyClient.evaluate() interface (a client-side method).
Evaluation uses a **dedicated server-only model instance (server_model)**
that belongs to no client — the 8 clients each already hold their own
model on the GPU, and the server now holds one more of its own (since this
is a local simulation sharing one process/GPU, not a real distributed
deployment, 9 model instances effectively coexist). At evaluation time,
that round's aggregated fit() results (global_state) are loaded into
server_model. Right after aggregation all clients are overwritten with the
same global parameters and see the same shared held-out set, so evaluating
it any number of times gives the same result (aside from floating-point
error) — hence the server evaluates this held-out set exactly once. Under
this design, per-client fairness variance is meaningless (all values are
identical) and is therefore not computed (see the README's Known
Limitations). ROUGE-L (requires generation, much more expensive) is also
computed by the server directly, not every round but exactly once after
the loop ends, on the final global_state, over the category-stratified
sample (rouge_eval_dataset) only.
"""

import json
import os
import time
from typing import List

import torch

from src.checkpointing import resume_or_start_fresh, save_checkpoint
from src.communication import compute_payload_bytes, get_trainable_state_dict, set_trainable_state_dict
from src.convergence import ConvergenceTracker
from src.models import get_model
from src.scaffold import scaffold_aggregate, scaffold_client_fit, zeros_like_trainable
from src.server_eval import evaluate_global_model, evaluate_global_model_generation


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

    # 서버 전용 모델 인스턴스 — 클라이언트 객체와 무관하게 평가만 전담한다
    # (지도교수 피드백: evaluate on server, not individual clients). 8개
    # 클라이언트도 각자 자기 모델을 GPU에 들고 있으므로, 이건 같은 프로세스
    # 안에 모델 인스턴스가 하나(서버) 더 추가되는 것일 뿐이다 — 실제
    # 분산 배포가 아니라 로컬 시뮬레이션이라 전부 같은 GPU/프로세스를 쓴다.
    # A server-only model instance — dedicated to evaluation, independent
    # of any client object (advisor feedback: evaluate on server, not
    # individual clients). Since the 8 clients each already hold their own
    # model on the GPU, this just adds one more model instance (the
    # server's) to the same process — this is a local simulation, not a
    # real distributed deployment, so everything shares the same GPU/process.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    server_model = get_model(config)
    server_model.to(device)

    start_round, resumed_state, extra = resume_or_start_fresh(checkpoint_base_dir, run_name)
    template_state = get_trainable_state_dict(server_model)
    global_state = resumed_state if resumed_state is not None else {k: v.clone() for k, v in template_state.items()}
    global_control = extra.get("global_control") or zeros_like_trainable(server_model)
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
            # Common weighted-average aggregation for FedAvg/FedProx.
            # FedProx's proximal term is already reflected in the loss inside
            # fit() in fl_client.py.
            total_ex = sum(r["num_examples"] for r in fit_results)
            keys = list(global_state.keys())
            new_state = {k: torch.zeros_like(global_state[k]) for k in keys}
            for r in fit_results:
                weight = r["num_examples"] / total_ex
                for i, k in enumerate(keys):
                    new_state[k] += weight * torch.tensor(r["params"][i])
            global_state = new_state

        # PPL 전용 평가: 서버 전용 모델(server_model)로 서버가 직접 수행
        # (위 모듈 docstring 참고 — 클라이언트 객체를 거치지 않고, 8명 다
        # 평가해도 결과가 동일해 1번만 함). eval_dataset은 클라이언트마다
        # 다른 게 아니라 전역 공유라 clients[0]에서 그냥 참조만 가져온다.
        # PPL-only evaluation: performed by the server directly, on the
        # server's own model instance (see the module docstring above —
        # no client object involved, and evaluating all 8 clients would
        # give the same result, so it's done only once). eval_dataset is
        # shared globally (not client-specific), so grabbing the reference
        # via clients[0] is just a convenience.
        set_trainable_state_dict(server_model, global_state)
        val_loss, _n = evaluate_global_model(server_model, clients[0].eval_dataset, device=device)

        # FedAvg/FedProx의 fit_results에만 metrics(peak_vram_gb 포함)가 있음 —
        # SCAFFOLD 경로는 scaffold_client_fit()이 이 계측을 하지 않아 빠짐.
        # FedAvg/FedProx's fit_results carry metrics (incl. peak_vram_gb) —
        # the SCAFFOLD path is absent since scaffold_client_fit() doesn't
        # perform this measurement.
        client_vram = [r["metrics"]["peak_vram_gb"] for r in fit_results if "metrics" in r]
        peak_vram_gb = sum(client_vram) / len(client_vram) if client_vram else None

        round_record = {
            "round": round_num,
            "val_loss": val_loss,
            "val_perplexity": float(torch.exp(torch.tensor(val_loss))),
            "round_latency_sec": time.perf_counter() - round_start,
            "communication_bytes_this_round": payload_bytes_per_round * len(selected_idx),
            "peak_vram_gb": peak_vram_gb,
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
    # 기준으로 ROUGE-L 생성 평가를 정확히 1회만 서버가 직접 수행 (Subtask 1.2
    # task performance 지표 중 생성이 필요한 부분 — 비용 통제를 위해
    # 라운드마다 돌리지 않는다).
    # After the loop ends (early stop on convergence or reaching num_rounds),
    # the server runs the ROUGE-L generation evaluation directly, exactly
    # once, on the final global parameters (the generation-requiring part of
    # the Subtask 1.2 task performance metrics — not run every round in
    # order to control cost).
    set_trainable_state_dict(server_model, global_state)
    final_gen = evaluate_global_model_generation(
        server_model, clients[0].tokenizer, clients[0].rouge_eval_dataset, device=device
    )
    if "rouge_l" in final_gen:
        round_records[-1]["rouge_l"] = final_gen["rouge_l"]
        with open(round_log_path) as f:
            lines = f.readlines()
        lines[-1] = json.dumps(round_records[-1]) + "\n"
        with open(round_log_path, "w") as f:
            f.writelines(lines)
        if wb is not None:
            import wandb
            wandb.log({"rouge_l": final_gen["rouge_l"]}, step=round_records[-1]["round"])

        gen_path = os.path.join(log_dir, f"{run_name}_generations.jsonl")
        predictions = final_gen.get("predictions", [])
        categories = final_gen.get("categories", [])
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
    total_latency_sec = sum(r["round_latency_sec"] for r in round_records)
    vram_values = [r["peak_vram_gb"] for r in round_records if r["peak_vram_gb"] is not None]
    avg_peak_vram_gb = sum(vram_values) / len(vram_values) if vram_values else None
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
        "total_latency_sec": total_latency_sec,
        "avg_peak_vram_gb": avg_peak_vram_gb,
        "round_records": round_records,
    }
