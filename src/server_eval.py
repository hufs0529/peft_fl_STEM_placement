"""서버 측 글로벌 모델 평가 — 지도교수 피드백("evaluation is based on the
test on server, not individual clients") 반영.

fl_runner.py(서버 루프)가 이 모듈의 함수를 직접 호출해 글로벌 모델을
평가한다 — Flower의 NumPyClient.evaluate() 인터페이스를 거치지 않는다.
클라이언트 객체(clients[0])의 모델을 재사용하는 건 이미 로드돼 있는
모델을 아끼기 위한 메모리 절약 디테일일 뿐, "클라이언트가 스스로를
평가"하는 것과는 다르다 — 실제로 평가를 수행하는 주체(호출부)는 서버
루프이고, 대상은 그 라운드의 fit() 결과를 집계(aggregate)한 뒤의
global_state다.

집계 직후 모든 클라이언트가 동일한 global 파라미터와 동일한 공유
held-out set을 갖게 되므로, 이 held-out을 몇 번 평가하든 결과는
동일하다(부동소수점 오차 제외) — 그래서 서버는 이 평가를 (global
파라미터가 로드된) 모델 인스턴스 하나로 딱 1번만 수행한다.

Server-side global-model evaluation — reflects advisor feedback
("evaluation is based on the test on server, not individual clients").

fl_runner.py (the server loop) calls this module's functions directly to
evaluate the global model — it does not go through Flower's
NumPyClient.evaluate() interface. Reusing a client object's already-loaded
model (clients[0].model) is purely a memory-saving detail, not "a client
evaluating itself" — the caller (the server loop) is what actually
performs the evaluation, and the target is global_state after that
round's fit() results have been aggregated.

Right after aggregation, every client holds identical global parameters
and sees the identical shared held-out set, so evaluating this held-out
set any number of times gives the same result (aside from floating-point
error) — hence the server performs this evaluation exactly once, using
one model instance loaded with the global parameters.
"""

from typing import Dict, Tuple

import torch

from src.metrics import compute_rouge_l, generate_responses


def evaluate_global_model(model, eval_dataset, device: str = "cuda") -> Tuple[float, int]:
    """held-out 전체에 대한 PPL 평가(teacher-forcing). (avg_loss, n) 반환.

    PPL evaluation (teacher-forcing) over the full held-out set. Returns
    (avg_loss, n)."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            batch = {k: item[k].unsqueeze(0).to(device) for k in ("input_ids", "attention_mask", "labels")}
            total_loss += model(**batch).loss.item()
            n += 1
    return total_loss / max(n, 1), n


def evaluate_global_model_generation(model, tokenizer, rouge_eval_dataset, device: str = "cuda") -> Dict:
    """카테고리 층화 샘플에 대한 ROUGE-L 생성 평가(최종 라운드 1회).

    ROUGE-L generation evaluation over the category-stratified sample
    (once, at the final round)."""
    model.eval()
    ds = rouge_eval_dataset
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    references = [ds[i]["reference_response"] for i in range(len(ds))]
    predictions = generate_responses(model, tokenizer, prompts, device=device)
    return {
        "rouge_l": compute_rouge_l(predictions, references),
        "predictions": predictions,
        "categories": [ds[i]["category"] for i in range(len(ds))],
    }
