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
#     PYTHON=/path/to/python bash scripts/run_alpha100.sh   # 인터프리터 직접 지정
#
# 체크포인트가 있으면 자동으로 이어서 실행되므로 중단 후 재실행해도 안전하다.
# Safe to re-run after an interruption: each run resumes from its checkpoint.

set -euo pipefail
cd "$(dirname "$0")/.."

ALPHA=100

# 의존성(transformers)이 설치된 파이썬을 찾는다. 가상환경을 활성화하지 않고
# 실행하면 시스템 python이 잡혀 ModuleNotFoundError가 나기 때문.
# Find a python that actually has the dependencies installed -- running
# without activating the virtualenv otherwise picks up the system python
# and fails with ModuleNotFoundError.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in venv/bin/python .venv/bin/python python3 python; do
    if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c "import transformers, peft, flwr" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
      fi
    fi
  done
fi

if [ -z "$PYTHON" ]; then
  echo "오류: transformers/peft/flwr를 import할 수 있는 python을 찾지 못했습니다." >&2
  echo "error: no python found with transformers/peft/flwr installed." >&2
  echo >&2
  echo "가상환경을 활성화하거나 인터프리터를 직접 지정하세요:" >&2
  echo "Activate the virtualenv, or point at the interpreter directly:" >&2
  echo "    source venv/bin/activate && bash scripts/run_alpha100.sh" >&2
  echo "    PYTHON=/path/to/venv/bin/python bash scripts/run_alpha100.sh" >&2
  exit 1
fi

echo "python: $PYTHON ($("$PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
echo

for compression in "lora" "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] peft=$compression  fl=$fl  alpha=$ALPHA"
    echo "=============================================================="
    "$PYTHON" scripts/run_experiment.py --peft $compression --fl "$fl" --alpha "$ALPHA"
  done
done

echo
echo "alpha=$ALPHA 6조합 완료. 24조합 분석 실행:"
echo "    $PYTHON scripts/analyze_interaction.py"
