"""Week 1 파일럿: 실제 모델(3B)로 FL 없이 단일 클라이언트 loss 감소만 확인.

사용법:
    python scripts/run_dev_pilot.py --stage single_client
"""

import argparse

import torch
import yaml

from src.models import get_model


def run_single_client_check(config: dict):
    model = get_model(config)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config["training"]["learning_rate"]
    )

    dummy_input = torch.randint(0, 1000, (2, 32))
    losses = []
    for step in range(5):
        output = model(input_ids=dummy_input, labels=dummy_input)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(output.loss.item())
        print(f"step {step}: loss={output.loss.item():.4f}")

    assert losses[-1] < losses[0], "loss가 감소하지 않음 — 학습 루프 점검 필요"
    print("single_client 검증 통과: loss가 감소함")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["single_client"], required=True)
    parser.add_argument("--config", default="configs/dev_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == "single_client":
        run_single_client_check(config)


if __name__ == "__main__":
    main()
