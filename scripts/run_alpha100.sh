#!/usr/bin/env bash
# 지도교수 피드백(2026-09-08): Dirichlet alpha=100(near-IID 끝점) 6조합 실행.
# 기존 18조합(alpha 0.1/1/10)에 더해 총 24조합이 된다.
#
# Advisor feedback (2026-09-08): run the 6 combinations at Dirichlet
# alpha=100 (the near-IID endpoint). Added to the existing 18 combinations
# (alpha 0.1/1/10), this makes 24 in total.
#
# 사용법 / Usage:
#     bash scripts/run_alpha100.sh
#
# 체크포인트가 있으면 자동으로 이어서 실행되므로 중단 후 재실행해도 안전하다.
# Safe to re-run after an interruption: each run resumes from its checkpoint.

set -euo pipefail
cd "$(dirname "$0")/.."

ALPHA=100

for compression in "lora" "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] peft=$compression  fl=$fl  alpha=$ALPHA"
    echo "=============================================================="
    python scripts/run_experiment.py --peft $compression --fl "$fl" --alpha "$ALPHA"
  done
done

echo
echo "alpha=$ALPHA 6조합 완료. 24조합 분석 실행:"
echo "    python scripts/analyze_interaction.py"
