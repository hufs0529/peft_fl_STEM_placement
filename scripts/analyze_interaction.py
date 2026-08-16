"""Week 5 (Subtask 1.3 / 2.1 선정) — core 6개 조합 완료 후 실행.

Task 1의 6개 조합(3 FL x 2 PEFT) 로그를 모두 읽어:
  1) Subtask 1.3: 상호작용 효과(FedProx/SCAFFOLD가 FedAvg 대비 개선하는 정도가
     LoRA -> QLoRA에서 얼마나 달라지는가) 계산 — Aim 1의 직접적인 답.
  2) Subtask 2.1: DoRA 진단 실험과 짝지을 FL 알고리즘(|상호작용|이 가장 큰 쪽) 결정.
  3) Subtask 2.2: local_epochs=5 강건성 점검에 쓸 "성능이 가장 좋은 core 조합" 결정.

사용법:
    python scripts/analyze_interaction.py
(results/logs/ 안의 core run 6개 *_rounds.jsonl을 모두 읽어 계산)

Week 5 (Subtask 1.3 / 2.1 selection) — run after the core 6 combinations
are complete.

Reads the logs for all 6 combinations of Task 1 (3 FL x 2 PEFT) to:
  1) Subtask 1.3: compute the interaction effect (how much the degree to
     which FedProx/SCAFFOLD improve over FedAvg changes going from
     LoRA -> QLoRA) — the direct answer to Aim 1.
  2) Subtask 2.1: decide the FL algorithm to pair with the DoRA diagnostic
     experiment (whichever has the larger |interaction|).
  3) Subtask 2.2: decide "the best-performing core combination" to use
     for the local_epochs=5 robustness check.

Usage:
    python scripts/analyze_interaction.py
(reads and computes over all 6 core run *_rounds.jsonl files in
results/logs/)
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (
    compute_interaction_effects,
    select_best_performing_combination,
    select_largest_interaction_fl_algorithm,
)

CORE_PEFTS = ["lora", "qlora"]
CORE_FLS = ["fedavg", "fedprox", "scaffold"]


def load_run_summary(peft: str, fl: str) -> dict:
    matches = glob.glob(f"results/logs/{peft}_{fl}_ep1_rounds.jsonl")
    if not matches:
        return None
    with open(matches[0]) as f:
        lines = [json.loads(line) for line in f]
    last = lines[-1]
    total_comm = sum(r["communication_bytes_this_round"] for r in lines)
    return {
        "peft": peft,
        "fl": fl,
        "run_name": matches[0].split("/")[-1].replace("_rounds.jsonl", ""),
        "final_val_loss": last["val_loss"],
        "val_perplexity": last["val_perplexity"],
        "rouge_l": last.get("rouge_l"),
        "total_communication_bytes": total_comm,
        "rounds_run": len(lines),
    }


def main():
    run_results = []
    for peft in CORE_PEFTS:
        for fl in CORE_FLS:
            summary = load_run_summary(peft, fl)
            if summary is None:
                print(f"경고: {peft}/{fl} 로그를 찾지 못함 — 아직 실행되지 않았을 수 있음")
                continue
            run_results.append(summary)

    if len(run_results) < 6:
        print(f"\n{len(run_results)}/6 core 조합만 완료됨. 전부 완료 후 다시 실행하세요.")
        return

    print("\n=== Core 6조합 요약 ===")
    for r in run_results:
        print(f"  {r['run_name']:20s} PPL={r['val_perplexity']:.4f}  comm={r['total_communication_bytes']:,}B")

    effects = compute_interaction_effects(run_results, performance_field="val_perplexity")
    print("\n=== Subtask 1.3: 상호작용 효과 (val_perplexity 기준, 양수=개선) ===")
    for peft in ("lora", "qlora"):
        e = effects["per_peft"][peft]
        print(f"  {peft:6s}  fedavg_ppl={e['fedavg']:.4f}  fedprox_delta={e['fedprox_delta']:+.4f}  scaffold_delta={e['scaffold_delta']:+.4f}")
    print(f"  interaction_fedprox  (qlora_delta - lora_delta) = {effects['interaction_fedprox']:+.4f}")
    print(f"  interaction_scaffold (qlora_delta - lora_delta) = {effects['interaction_scaffold']:+.4f}")
    print("  (0에 가까울수록 QLoRA에서도 교정 효과가 그대로 유지됨을 의미)")

    # rounds_run은 이미 매 라운드 무료로 로깅되는 값이라 추가 계산 없이 재사용.
    # FedProx/SCAFFOLD가 존재하는 이유 자체가 "더 빨리/안정적으로 수렴시키는 것"이라
    # 이 연구질문에 val_perplexity보다 오히려 더 직접적인 신호가 될 수 있다.
    # rounds_run is already logged for free every round, so it is reused
    # here with no extra computation. The very reason FedProx/SCAFFOLD
    # exist is "to converge faster/more stably," so for this research
    # question it can actually be a more direct signal than
    # val_perplexity.
    conv_effects = compute_interaction_effects(run_results, performance_field="rounds_run")
    print("\n=== Subtask 1.3 (수렴 속도 기준, rounds_run — 양수=더 빨리 수렴) ===")
    for peft in ("lora", "qlora"):
        e = conv_effects["per_peft"][peft]
        print(f"  {peft:6s}  fedavg_rounds={e['fedavg']:.1f}  fedprox_delta={e['fedprox_delta']:+.2f}  scaffold_delta={e['scaffold_delta']:+.2f}")
    print(f"  interaction_fedprox  (qlora_delta - lora_delta) = {conv_effects['interaction_fedprox']:+.2f}")
    print(f"  interaction_scaffold (qlora_delta - lora_delta) = {conv_effects['interaction_scaffold']:+.2f}")
    print("  (val_perplexity 기준과 방향이 다르면 두 지표가 서로 다른 것을 포착했다는 뜻 — 리포트에서 함께 논의)")

    largest_fl = select_largest_interaction_fl_algorithm(effects)
    print(f"\n>>> Subtask 2.1: DoRA를 짝지을 FL 알고리즘 = {largest_fl} <<<")
    print(f"    python scripts/run_experiment.py --peft dora --fl {largest_fl}")

    best = select_best_performing_combination(run_results, performance_field="val_perplexity")
    print(f"\n>>> Subtask 2.2: local_epochs=5 강건성 점검 대상 = {best['run_name']} <<<")
    print(f"    python scripts/run_experiment.py --peft {best['peft']} --fl {best['fl']} --local-epochs 5")


if __name__ == "__main__":
    main()
