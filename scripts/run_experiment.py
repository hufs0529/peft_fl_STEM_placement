"""Week 4 본 실험 실행 진입점.

Task 1 core (Subtask 1.2, 6회 실행):
    python scripts/run_experiment.py --peft lora   --fl fedavg
    python scripts/run_experiment.py --peft lora   --fl fedprox
    python scripts/run_experiment.py --peft lora   --fl scaffold
    python scripts/run_experiment.py --peft qlora  --fl fedavg
    python scripts/run_experiment.py --peft qlora  --fl fedprox
    python scripts/run_experiment.py --peft qlora  --fl scaffold

6개가 모두 끝나면 scripts/analyze_interaction.py를 실행해 Subtask 1.3
상호작용 효과와, 아래 Task 2 진단 실험에 쓸 조합을 확인한다.

Task 2 diagnostic:
    # Subtask 2.1 — DoRA를 <largest_interaction_fl>과 짝지어 양자화 vs 압축 분리
    python scripts/run_experiment.py --peft dora --fl <largest_interaction_fl>

    # Subtask 2.2 — 성능이 가장 좋았던 core 조합을 local_epochs=5로 재실행
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --local-epochs 5

주의: 실행 전 반드시 pytest tests/ 로 검증을 통과시킬 것.

Entry point for running the Week 4 main experiment.

Task 1 core (Subtask 1.2, run 6 times):
    python scripts/run_experiment.py --peft lora   --fl fedavg
    python scripts/run_experiment.py --peft lora   --fl fedprox
    python scripts/run_experiment.py --peft lora   --fl scaffold
    python scripts/run_experiment.py --peft qlora  --fl fedavg
    python scripts/run_experiment.py --peft qlora  --fl fedprox
    python scripts/run_experiment.py --peft qlora  --fl scaffold

Once all 6 are finished, run scripts/analyze_interaction.py to check the
Subtask 1.3 interaction effects and the combination to use for the Task 2
diagnostic experiments below.

Task 2 diagnostic:
    # Subtask 2.1 — pair DoRA with <largest_interaction_fl> to isolate
    # quantization vs. compression
    python scripts/run_experiment.py --peft dora --fl <largest_interaction_fl>

    # Subtask 2.2 — rerun the best-performing core combination with
    # local_epochs=5
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --local-epochs 5

지도교수 피드백 — 압축률/Dirichlet 강건성 스윕:
    core 6조합 중 상호작용이 가장 컸던 조합(<best_peft>/<best_fl>) 하나에 한해,
    --alpha와 --qlora-bits로 α와 압축 비트폭을 스윕한다. 새 값이 core 기본값
    (α=0.5, 4bit)과 다르면 run_name에 접미사가 붙어 core 결과를 덮어쓰지 않는다.

    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha 0.1
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha 1.0
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --qlora-bits 8

Advisor feedback — compression-rate / Dirichlet robustness sweep:
    For the single combination with the largest interaction among the core
    6 (<best_peft>/<best_fl>), sweep alpha and the quantization bit-width
    with --alpha and --qlora-bits. When the new value differs from the
    core default (alpha=0.5, 4-bit), a suffix is appended to run_name so
    the core results are never overwritten.

    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha 0.1
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --alpha 1.0
    python scripts/run_experiment.py --peft <best_peft> --fl <best_fl> --qlora-bits 8

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
    # alpha/qlora_bits가 core 기본값(0.5, 4bit)과 같으면 기존 run_name 형식을
    # 그대로 유지 — analyze_interaction.py의 core 6조합 파일명 매칭과의
    # 하위 호환성을 위함. 기본값과 다를 때만(강건성 스윕) 접미사를 붙여
    # core 결과를 덮어쓰지 않도록 함.
    #
    # If alpha/qlora_bits equal the core defaults (0.5, 4-bit), the run_name
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
    if alpha != 0.5:
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
        help="지도교수 피드백: Dirichlet 비IID 강도 스윕 (기본 0.5, 예: 0.1/1.0) — "
             "최적 조합 1개에만 적용 권장 / advisor feedback: Dirichlet non-IID "
             "intensity sweep (default 0.5, e.g. 0.1/1.0) — recommended only for "
             "the single best-performing combination",
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
