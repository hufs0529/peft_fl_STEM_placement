"""Week 4 본 실험 실행 진입점 (Week 5 지도교수 피드백으로 설계 개정).

Core design (개정): 압축률(3) x FL(2) x Dirichlet alpha(3) = 18회 실행.
지도교수 피드백: 이 연구의 중점을 "압축률과 non-IID 강도의 상관관계"로
재설정 — SCAFFOLD는 core 설계에서 제외(코드/테스트는 그대로 유지).

    for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
      for fl in fedavg fedprox; do
        for alpha in 0.1 1 10; do
          python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
        done
      done
    done

18개가 모두 끝나면 scripts/analyze_interaction.py를 실행해 압축률 x alpha
상관관계와, Task 2 진단 실험(local_epochs=5)에 쓸 조합을 확인한다.

Task 2 diagnostic:
    # Subtask 2.2 — 성능이 가장 좋았던 조합을 local_epochs=5로 재실행
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha <best_alpha> [--qlora-bits 8] --local-epochs 5

주의: 실행 전 반드시 pytest tests/ 로 검증을 통과시킬 것.

Entry point for running the Week 4 main experiment (design revised in
Week 5 per advisor feedback).

Core design (revised): compression (3) x FL (2) x Dirichlet alpha (3) =
18 runs. Advisor feedback reframed this project's focus as "the
correlation between compression rate and non-IID intensity" — SCAFFOLD
is dropped from the core design (its code/tests are kept).

    for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
      for fl in fedavg fedprox; do
        for alpha in 0.1 1 10; do
          python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
        done
      done
    done

Once all 18 are finished, run scripts/analyze_interaction.py to check the
compression x alpha correlation and the combination to use for the Task 2
diagnostic experiment below.

Task 2 diagnostic:
    # Subtask 2.2 — rerun the best-performing combination with local_epochs=5
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha <best_alpha> [--qlora-bits 8] --local-epochs 5

Note: be sure to pass pytest tests/ before running.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from transformers import AutoTokenizer

from src.data import (
    build_category_tagged_holdout,
    build_client_dataloaders,
    build_holdout_split,
    load_raw_dolly15k,
    sample_for_generation_eval,
)
from src.fl_client import FlowerClient
from src.fl_runner import run_federated_training
from src.partitioning import partition_by_category


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    for key, value in overrides.items():
        if value is not None:
            section, field = key.split(".")
            config[section][field] = value
    return config


def build_run_name(config: dict) -> str:
    # alpha/qlora_bits가 core 기본값(1, 4bit)과 같으면 기존 run_name 형식을
    # 그대로 유지 — analyze_interaction.py의 core 6조합 파일명 매칭과의
    # 하위 호환성을 위함. 기본값과 다를 때만(강건성 스윕) 접미사를 붙여
    # core 결과를 덮어쓰지 않도록 함.
    #
    # If alpha/qlora_bits equal the core defaults (1, 4-bit), the run_name
    # format is left unchanged — this keeps backward compatibility with
    # analyze_interaction.py's filename matching for the core 6
    # combinations. A suffix is appended only when they differ from the
    # defaults (i.e. a robustness sweep run), so sweep runs never overwrite
    # the core results.
    name = (
        f"{config['peft']['type']}_{config['fl_algorithm']['type']}"
        f"_ep{config['federated']['local_epochs']}"
    )
    alpha = config["partitioning"]["alpha"]
    if alpha != 1:
        name += f"_a{alpha}"
    if config["peft"]["type"] == "qlora":
        qlora_bits = config["peft"].get("qlora_bits", 4)
        if qlora_bits != 4:
            name += f"_q{qlora_bits}bit"
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--peft", choices=["lora", "qlora", "dora"], required=True)
    parser.add_argument("--fl", choices=["fedavg", "fedprox", "scaffold"], required=True)
    parser.add_argument("--local-epochs", type=int, default=None, help="Subtask 2.2 강건성 점검: 1(기본) vs 5")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="지도교수 피드백: Dirichlet 비IID 강도. Core 18조합 설계는 0.1/1/10 "
             "전부를 사용 / advisor feedback: Dirichlet non-IID intensity. The core "
             "18-combination design uses all of 0.1/1/10",
    )
    parser.add_argument(
        "--qlora-bits", type=int, default=None, choices=[4, 8],
        help="지도교수 피드백: QLoRA 압축 비트폭 스윕 (기본 4, qlora에만 해당) / "
             "advisor feedback: QLoRA compression bit-width sweep (default 4, "
             "qlora only)",
    )
    args = parser.parse_args()

    overrides = {
        "peft.type": args.peft,
        "fl_algorithm.type": args.fl,
        "federated.local_epochs": args.local_epochs,
        "partitioning.alpha": args.alpha,
        "peft.qlora_bits": args.qlora_bits,
    }
    config = load_config(args.config, overrides)
    run_name = build_run_name(config)

    print(f"=== {run_name} ===")
    print("Loading Dolly-15k ...")
    raw = load_raw_dolly15k()
    train_pool, holdout_eval = build_holdout_split(raw, holdout_fraction=config["data"]["holdout_fraction"])

    client_data = partition_by_category(
        train_pool,
        num_clients=config["federated"]["num_clients"],
        alpha=config["partitioning"]["alpha"],
        min_category_threshold=config["partitioning"]["min_category_threshold"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaders = build_client_dataloaders(
        client_data, holdout_eval, tokenizer,
        max_length=config["data"]["max_seq_length"],
        micro_batch_size=config["training"]["micro_batch_size"],
    )
    holdout_dataset = build_category_tagged_holdout(holdout_eval, tokenizer, config["data"]["max_seq_length"])

    # ROUGE-L 생성 평가는 held-out 전체가 아니라 카테고리 층화 샘플만 사용
    # (자기회귀 생성이 비싸므로 — Subtask 1.2, config['data']['rouge_l_sample_size']).
    # ROUGE-L generation evaluation uses only a category-stratified sample,
    # not the full held-out set (since autoregressive generation is
    # expensive — Subtask 1.2, config['data']['rouge_l_sample_size']).
    rouge_sample_size = config["data"].get("rouge_l_sample_size", len(holdout_eval))
    rouge_eval_raw = sample_for_generation_eval(holdout_eval, sample_size=rouge_sample_size)
    rouge_eval_dataset = build_category_tagged_holdout(rouge_eval_raw, tokenizer, config["data"]["max_seq_length"])

    clients = [
        FlowerClient(config, cid, loaders[cid]["train"], holdout_dataset, tokenizer, rouge_eval_dataset=rouge_eval_dataset)
        for cid in range(config["federated"]["num_clients"])
    ]

    result = run_federated_training(config, clients, run_name)

    print(f"\n=== {run_name} 완료 ===")
    print(f"실행 라운드: {result['rounds_run']} (수렴 라운드: {result['converged_round']})")
    print(f"최종 val_loss: {result['final_val_loss']:.4f} / PPL: {result['val_perplexity']:.4f}")
    if result.get("rouge_l") is not None:
        print(f"ROUGE-L(최종 라운드, 샘플 {rouge_sample_size}개): {result['rouge_l']:.4f}")
    print(f"총 통신비용(bytes): {result['total_communication_bytes']:,}")


if __name__ == "__main__":
    main()
