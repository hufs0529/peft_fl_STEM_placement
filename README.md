# Dolly-15k FL × PEFT Research Placement — Week 1 진행 보고 (Week 1 Progress Report)

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가? (`Does Drift-Correction Sophistication Interact with PEFT
Quantisation Noise in Federated LLM Fine-Tuning?`)

> 이 README는 **Week 1 시점까지 완료된 것만** 반영합니다. 이후 주차 브랜치
> (`week2`, `week3`, ...)로 갈수록 이 문서도 함께 누적 갱신됩니다.

**Research Question**: Does the benefit of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist even as PEFT quantization (LoRA→QLoRA) intensifies? (`Does Drift-Correction
Sophistication Interact with PEFT Quantisation Noise in Federated LLM Fine-Tuning?`)

> This README reflects **only what was completed as of Week 1**. As later weekly
> branches (`week2`, `week3`, ...) are added, this document will be cumulatively
> updated along with them.

---

## 진행 상황 요약 (Progress Summary)

| 주차 | 목표 | 상태 |
|---|---|---|
| **Week 1** | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA), 단일 클라이언트 loss 감소 | ✅ 완료 (이 브랜치) |
| Week 2 | Dolly-15k 파이프라인 + Dirichlet 파티셔닝 검증 | 예정 (`week2` 브랜치) |
| Week 3 | FedProx/SCAFFOLD 통합 + 체크포인트/수렴 기준 | 예정 |
| Week 4 | Core 6조합 실행 + DoRA/local-epoch 진단 | 예정 |
| Week 5 | 상호작용 분석 + 카테고리별/수렴궤적 진단 | 예정 |
| Week 6 | 최종 리포트 | 예정 |

| Week | Goal | Status |
|---|---|---|
| **Week 1** | Verify model + GPU environment, PEFT wiring (LoRA/QLoRA/DoRA), single-client loss reduction | ✅ Done (this branch) |
| Week 2 | Dolly-15k pipeline + Dirichlet partitioning verification | Planned (`week2` branch) |
| Week 3 | FedProx/SCAFFOLD integration + checkpoint/convergence criteria | Planned |
| Week 4 | Run Core 6 combinations + DoRA/local-epoch diagnostics | Planned |
| Week 5 | Interaction analysis + per-category/convergence-trajectory diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Week 1에서 한 일 (What Was Done in Week 1)

### 1. 프로젝트 스켈레톤 (Project Skeleton)
- `requirements.txt`: torch, transformers, peft, bitsandbytes, flwr[simulation] 등
- `configs/dev_config.yaml`: Week 1~3 검증용 미니어처 설정

- `requirements.txt`: torch, transformers, peft, bitsandbytes, flwr[simulation], etc.
- `configs/dev_config.yaml`: miniature configuration for Week 1-3 verification

### 2. PEFT 배선 (PEFT Wiring) (`src/models.py`)
연구계획서 Subtask 1.1(core: LoRA/QLoRA)과 Subtask 2.1(진단: DoRA)에 필요한
3가지 PEFT 방식을 하나의 `get_model(config)` 함수로 통합:

- **LoRA**: r=8, α=16, dropout=0.05, `target_modules=["q_proj","v_proj"]`
- **QLoRA**: 위와 동일한 LoRA 설정 + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- **DoRA**: LoRA와 압축률은 같지만 양자화가 없는 진단 대조군(`use_dora=True`) — Task 2에서
  "상호작용이 양자화 때문인지 압축 자체 때문인지"를 가르는 데 씀
- 정의되지 않은 PEFT type은 `ValueError`

Integrates the 3 PEFT methods required by Subtask 1.1 (core: LoRA/QLoRA) and
Subtask 2.1 (diagnostic: DoRA) of the research proposal into a single
`get_model(config)` function:

- **LoRA**: r=8, α=16, dropout=0.05, `target_modules=["q_proj","v_proj"]`
- **QLoRA**: the same LoRA settings as above + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- **DoRA**: a diagnostic control with the same compression ratio as LoRA but
  no quantization (`use_dora=True`) — used in Task 2 to separate "whether the
  interaction is due to quantization or to compression itself"
- An undefined PEFT type raises `ValueError`

### 3. FL 파라미터 헬퍼 (FL Parameter Helper) (`src/communication.py`)
아직 Week 2 스코프인 "라운드트립" 자체는 테스트하지 않지만, 학습 가능한
파라미터만 뽑아내는 `get_trainable_state_dict()`가 이번 주 wiring 테스트에서
바로 필요해서 먼저 들여왔습니다.

The "round-trip" itself, which is still Week 2 scope, is not tested yet, but
`get_trainable_state_dict()`, which extracts only the trainable parameters,
was needed right away for this week's wiring tests, so it was introduced
first.

### 4. 검증 (Verification)
- `tests/test_model_wiring.py`: LoRA/DoRA에 학습 가능한 파라미터가 잡히는지,
  forward pass가 에러 없이 도는지, 정의 안 된 PEFT type이 `ValueError`를
  내는지 확인 (QLoRA는 GPU 환경에서만 테스트 — `bitsandbytes` 4bit 제약)
- `scripts/run_dev_pilot.py`: 실제 모델로 단일 클라이언트 5스텝 학습해서
  loss가 실제로 감소하는지 확인

- `tests/test_model_wiring.py`: checks that LoRA/DoRA capture trainable
  parameters, that the forward pass runs without error, and that an
  undefined PEFT type raises `ValueError` (QLoRA is tested only in a GPU
  environment — due to `bitsandbytes` 4bit constraints)
- `scripts/run_dev_pilot.py`: trains a single client for 5 steps with the
  actual model to verify that loss actually decreases

### 설계 노트: dev 모델 선택 (Design Note: Choosing the Dev Model)
처음엔 `gpt2`를 dev 모델로 쓰려 했으나, GPT-2의 어텐션은 결합형 `c_attn`
Conv1D라 `target_modules=["q_proj","v_proj"]`와 맞지 않아 LoRA 주입이
실패합니다. 본실험 모델(Qwen2.5-3B)과 **동일한 q_proj/v_proj 구조**를 가진
`hf-internal-testing/tiny-random-LlamaForCausalLM`으로 교체해 이 문제를
해결했습니다 — 작지만 구조가 같은 모델을 써야 Week 1의 wiring 검증이
실제로 의미가 있습니다.

Initially, `gpt2` was intended as the dev model, but GPT-2's attention uses
a fused `c_attn` Conv1D, which doesn't match `target_modules=["q_proj","v_proj"]`,
so LoRA injection fails. This was resolved by switching to
`hf-internal-testing/tiny-random-LlamaForCausalLM`, which has the **same
q_proj/v_proj structure** as the main experiment model (Qwen2.5-3B) — using
a small model with the same structure is what makes the Week 1 wiring
verification actually meaningful.

---

## 실행 방법 (How to Run)

```bash
pip install -r requirements.txt
pytest tests/test_model_wiring.py -v
python scripts/run_dev_pilot.py --stage single_client
```

**테스트 결과**: 4 passed, 1 skipped(QLoRA, GPU 필요)

**Test Results**: 4 passed, 1 skipped (QLoRA, GPU required)

---

## 다음 주 (Week 2) 예고 (Preview of Next Week — Week 2)

- `src/data.py`: Dolly-15k 로딩/토큰화/held-out 분리
- `src/partitioning.py`: Dirichlet(α=0.5) 비IID 파티셔닝
- `tests/test_roundtrip.py`, `tests/test_partitioning.py`

→ `week2` 브랜치 참고.

- `src/data.py`: Dolly-15k loading/tokenization/held-out split
- `src/partitioning.py`: Dirichlet (α=0.5) non-IID partitioning
- `tests/test_roundtrip.py`, `tests/test_partitioning.py`

→ See the `week2` branch.
