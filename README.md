# Dolly-15k FL × PEFT Research Placement — Week 3 진행 보고

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

> 이 README는 **Week 1~3 누적** 상태를 반영합니다.

---

## 진행 상황 요약

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA) | ✅ 완료 |
| Week 2 | Dolly-15k 파이프라인, 파라미터 왕복, Dirichlet(α=0.5) 파티셔닝 | ✅ 완료 |
| **Week 3** | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준, FL 통합 테스트 | ✅ 완료 (이 브랜치) |
| Week 4 | Core 6조합 실행 + DoRA/local-epoch 진단 | 예정 (`week4`) |
| Week 5 | 상호작용 분석 + 카테고리별/수렴궤적 진단 | 예정 |
| Week 6 | 최종 리포트 | 예정 |

---

## Week 1~2 요약

- **Week 1**: `src/models.py`(LoRA/QLoRA/DoRA 배선), `src/communication.py`(파라미터 헬퍼),
  wiring 테스트, dev pilot 스크립트
- **Week 2**: `src/data.py`(Dolly-15k 로딩/토큰화/held-out), `src/partitioning.py`
  (Dirichlet α=0.5 비IID 분배), roundtrip/partitioning 테스트

## Week 3에서 한 일 — 이번 주가 가장 큰 덩어리입니다

### 1. 학습/평가 지표 유틸 (`src/metrics.py`)
`track_vram_and_latency()`(학습 루프를 감싸는 context manager로 Peak VRAM·
Training Latency 측정), `compute_rouge_l()`, `generate_responses()`,
`count_trainable_parameters()` — 이후 클라이언트 구현이 바로 가져다 씁니다.

### 2. 수렴 기준 (`src/convergence.py`)
`ConvergenceTracker`: 최근 3라운드 연속 개선률이 1% 미만이면 수렴 판정.
FL 루프에 배선하기 전에 로직만 독립적으로 먼저 검증했습니다.

### 3. 체크포인트 (`src/checkpointing.py`)
라운드마다 저장(`save_checkpoint`), 재시작 시 자동 재개
(`resume_or_start_fresh`) — 스팟 인스턴스 회수·세션 끊김 대응.

### 4. FL 클라이언트 (`src/fl_client.py`)
- FedProx일 때만 `(μ/2)·‖local-global‖²` proximal term을 loss에 더함(μ=0.01)
- 첫 라운드에만 linear warmup
- `evaluate()`는 PPL(teacher-forcing 순전파)은 항상 계산하고, ROUGE-L(생성
  필요, 비쌈)은 별도의 `rouge_eval_dataset`(더 작은 샘플)에 대해서만,
  명시적으로 요청됐을 때만 계산

### 5. SCAFFOLD (`src/scaffold.py`)
Karimireddy et al. (2020) Option II의 **간소화 근사** 구현. control variate
보정을 그래디언트에 직접 더하는 방식(`scaffold_client_fit`)과, 전역
모델·전역 control variate를 함께 갱신하는 집계(`scaffold_aggregate`)로
구성 — FedAvg/FedProx의 weighted-average 집계와는 별도 경로.

### 6. 수동 FL 라운드 루프 (`src/fl_runner.py`)
`flwr.simulation.start_simulation()`은 라운드 중간 조기종료/재개가
어려워서, 집계·평가·체크포인트·수렴판정을 직접 구현한 루프를 씁니다:

```
fit(8클라이언트) → 집계(FedAvg/FedProx 평균 or SCAFFOLD 경로)
  → PPL 평가(대표 클라이언트 1개) → 체크포인트 저장 → 수렴 판정 → (조기종료 or 다음 라운드)
```

ROUGE-L 생성은 라운드마다가 아니라 **루프가 끝난 뒤 최종 파라미터로 딱 1번만**
계산합니다 — 자기회귀 생성이 훨씬 비싸기 때문입니다.

### 7. 검증
- `test_convergence.py`, `test_checkpointing.py`: 각 유닛 단독 검증
- `test_fl_integration.py`: FedAvg/FedProx/SCAFFOLD **세 경로 모두** 전체
  루프(fit→집계→평가→체크포인트→수렴판정)를 에러 없이 통과하는지,
  SCAFFOLD의 `local_control`이 라운드 간 유지되는지, FedProx의 proximal
  term이 실제로 동작하는지

### 설계 노트: 왜 PPL을 클라이언트 1개로만 재는가
집계 직후 8개 클라이언트는 전부 **동일한 global 파라미터**로 덮어써지고
**동일한 공용 held-out set**을 봅니다 — 같은 모델+같은 데이터로 8번
순전파해도 결과는 항상 같습니다(부동소수점 오차 제외). 그래서 대표
클라이언트 1개만 평가하고, 나머지 7번의 중복 계산은 하지 않습니다.
(반대로 `fit()`은 클라이언트마다 로컬 데이터가 달라 8번 다 필요합니다.)

---

## 실행 방법

```bash
pytest tests/test_convergence.py tests/test_checkpointing.py tests/test_fl_integration.py -v
```

**테스트 결과**: 18 passed, 1 skipped(QLoRA, GPU 필요) — Week 1~3 테스트 전체 통과.

---

## 다음 주 (Week 4) 예고

- `configs/experiment_config.yaml`: Qwen2.5-3B, 8클라이언트, 최대 10라운드 본실험 설정
- `src/evaluate.py`: 상호작용 효과 계산(`compute_interaction_effects`) — 이 프로젝트의
  핵심 산출물
- `scripts/run_experiment.py`: core 6조합 + Task 2 진단 실행 CLI
- `scripts/analyze_interaction.py`: 결과 분석 + DoRA/local-epoch 대상 자동 선정

→ `week4` 브랜치 참고.
