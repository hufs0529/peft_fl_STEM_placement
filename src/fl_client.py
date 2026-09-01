"""공통 클라이언트 — 계획서 v2 §4.1(warmup), §4.2(FedProx proximal term),
§5.3(VRAM/latency/ROUGE-L) 전부 통합된 버전.

evaluate()는 Flower의 NumPyClient 인터페이스를 만족시키기 위해 남아있을
뿐, 실제 평가는 지도교수 피드백("evaluation is based on the test on
server, not individual clients")에 따라 src/server_eval.py를 통해
서버 루프(fl_runner.py)가 직접 수행한다 — 자세한 내용은 그 모듈의
docstring 참고.

Common client — the version that integrates plan v2 §4.1 (warmup),
§4.2 (FedProx proximal term), and §5.3 (VRAM/latency/ROUGE-L) all together.

evaluate() is kept only to satisfy Flower's NumPyClient interface — actual
evaluation is performed directly by the server loop (fl_runner.py) via
src/server_eval.py, per advisor feedback ("evaluation is based on the
test on server, not individual clients"); see that module's docstring for
details.
"""

from typing import Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from transformers import get_linear_schedule_with_warmup

from src.communication import (
    get_trainable_state_dict,
    ndarrays_to_state_dict,
    set_trainable_state_dict,
    state_dict_to_ndarrays,
)
from src.metrics import count_trainable_parameters, track_vram_and_latency
from src.models import get_model
from src.server_eval import evaluate_global_model, evaluate_global_model_generation


class FlowerClient(NumPyClient):
    def __init__(self, config: dict, client_id: int, train_loader, eval_dataset, tokenizer, rouge_eval_dataset=None):
        self.config = config
        self.client_id = client_id
        self.model = get_model(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.train_loader = train_loader
        self.eval_dataset = eval_dataset
        # PPL은 held-out 전체(eval_dataset)로 매 라운드 계산하지만, ROUGE-L은
        # 생성이 필요해 훨씬 비싸므로 별도의 (더 작은, 카테고리 층화 샘플)
        # 데이터셋을 쓴다. 안 넘기면 eval_dataset과 동일하게 동작(하위 호환).
        # PPL is computed every round over the full held-out set (eval_dataset),
        # but ROUGE-L requires generation and is much more expensive, so it uses
        # a separate (smaller, category-stratified sample) dataset. If not
        # passed, it behaves the same as eval_dataset (backward compatible).
        self.rouge_eval_dataset = rouge_eval_dataset if rouge_eval_dataset is not None else eval_dataset
        self.tokenizer = tokenizer
        self.param_keys = list(get_trainable_state_dict(self.model).keys())

    def get_parameters(self, cfg) -> NDArrays:
        return state_dict_to_ndarrays(get_trainable_state_dict(self.model))

    def set_parameters(self, parameters: NDArrays):
        state_dict = ndarrays_to_state_dict(self.param_keys, parameters)
        set_trainable_state_dict(self.model, state_dict)

    def fit(self, parameters: NDArrays, cfg) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        lr = self.config["training"]["learning_rate"]
        accum_steps = self.config["training"]["grad_accumulation_steps"]
        local_epochs = self.config["federated"].get("local_epochs", 1)
        server_round = cfg.get("server_round", 1)

        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=lr)

        # 계획서 §4.1: 첫 라운드에만 linear warmup
        # Plan §4.1: linear warmup only on the first round
        total_steps = max(1, len(self.train_loader) // accum_steps) * local_epochs
        scheduler = None
        if server_round == 1:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
            )

        # 계획서 §4.2: FedProx proximal term (mu=0.01)
        # Plan §4.2: FedProx proximal term (mu=0.01)
        is_fedprox = self.config["fl_algorithm"]["type"] == "fedprox"
        proximal_mu = self.config["fl_algorithm"].get("proximal_mu", 0.01)
        global_ref = None
        if is_fedprox:
            global_ref = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

        self.model.train()
        num_examples = 0
        loss_log = []

        with track_vram_and_latency(self.device) as perf_stats:
            for _ in range(local_epochs):
                for step, batch in enumerate(self.train_loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    loss = self.model(**batch).loss / accum_steps

                    if is_fedprox:
                        prox_term = sum(
                            (param - global_ref[name]).pow(2).sum()
                            for name, param in self.model.named_parameters()
                            if param.requires_grad
                        )
                        loss = loss + (proximal_mu / 2) * prox_term

                    loss.backward()
                    if (step + 1) % accum_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                    loss_log.append(loss.item() * accum_steps)
                    num_examples += batch["input_ids"].shape[0]

        param_stats = count_trainable_parameters(self.model)
        avg_train_loss = sum(loss_log) / max(len(loss_log), 1)

        metrics = {
            "client_id": self.client_id,
            "train_loss": avg_train_loss,
            "peak_vram_gb": perf_stats["peak_vram_gb"],
            "latency_sec": perf_stats["latency_sec"],
            "trainable_params": param_stats["trainable_params"],
            "trainable_pct": param_stats["trainable_pct"],
        }
        return self.get_parameters(cfg), num_examples, metrics

    def evaluate(self, parameters: NDArrays, cfg) -> Tuple[float, int, Dict[str, Scalar]]:
        """Flower의 NumPyClient 인터페이스를 만족시키기 위해 남겨둔
        메서드다 — 실제 서버 루프(fl_runner.py)는 지도교수 피드백
        ("evaluation is based on the test on server, not individual
        clients")에 따라 이 메서드를 거치지 않고
        src/server_eval.py의 함수를 직접 호출한다. 로직 자체는 거기
        있는 걸 그대로 재사용해 중복을 피한다.

        Kept only to satisfy Flower's NumPyClient interface — the actual
        server loop (fl_runner.py) does not call this; per advisor
        feedback ("evaluation is based on the test on server, not
        individual clients"), it calls src/server_eval.py's functions
        directly. The logic here is reused from there, not duplicated.
        """
        self.set_parameters(parameters)
        run_generation = cfg.get("run_generation_metrics", False)

        avg_loss, n = evaluate_global_model(self.model, self.eval_dataset, device=self.device)
        metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }

        if run_generation:
            # rouge_eval_dataset은 eval_dataset(PPL 전체)보다 훨씬 작은
            # 카테고리 층화 샘플 — 생성 비용을 통제하기 위함(Subtask 1.2).
            # rouge_eval_dataset is a category-stratified sample that is much
            # smaller than eval_dataset (full PPL set) — this is to control
            # generation cost (Subtask 1.2).
            gen = evaluate_global_model_generation(self.model, self.tokenizer, self.rouge_eval_dataset, device=self.device)
            metrics["rouge_l"] = gen["rouge_l"]
            metrics["_per_example_categories"] = gen["categories"]
            metrics["_generated_predictions"] = gen["predictions"]

        return avg_loss, n, metrics
