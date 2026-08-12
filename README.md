# Dolly-15k FL × PEFT Research Placement — Week 4 진행 보고

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

> 이 README는 **Week 1~4 누적** 상태를 반영합니다.

---

## 진행 상황 요약

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA) | ✅ 완료 |
| Week 2 | Dolly-15k 파이프라인, Dirichlet(α=0.5) 파티셔닝 | ✅ 완료 |
| Week 3 | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준 | ✅ 완료 |
| **Week 4** | Core 6조합 실행 스크립트 + 상호작용 분석 + Task 2 진단 스크립트 | ✅ 완료 (이 브랜치) |
| Week 5 | 상호작용 분석 테스트 + 카테고리별/수렴궤적 진단 | 예정 (`week5`) |
| Week 6 | 최종 리포트 | 예정 |

---

## Week 1~3 요약
- **Week 1**: PEFT 배선(LoRA/QLoRA/DoRA), 파라미터 헬퍼, wiring 테스트
- **Week 2**: Dolly-15k 파이프라인, Dirichlet(α=0.5) 파티셔닝
- **Week 3**: FedProx/SCAFFOLD, 체크포인트, 수렴 기준, 수동 FL 라운드 루프,
  FL 통합 테스트(18 passed)

## Week 4에서 한 일

### 1. 본실험 설정 (`configs/experiment_config.yaml`)
Qwen2.5-3B-Instruct, 8클라이언트, α=0.5, 최대 10라운드, effective batch 16.
`rouge_l_sample_size`로 생성 평가 비용 통제.

### 2. 상호작용 효과 계산 (`src/evaluate.py`) — **이 프로젝트의 핵심 산출물**
- `compute_interaction_effects(run_results, performance_field)`: 각 PEFT(LoRA/QLoRA)에서
  FedProx/SCAFFOLD가 FedAvg 대비 얼마나 개선하는지(delta)를 구하고, 그 delta가
  LoRA→QLoRA로 갈 때 얼마나 달라지는지(`interaction = delta_qlora - delta_lora`)를
  계산. `performance_field`를 바꿔서 `val_perplexity`(성능)와 `rounds_run`(수렴 속도,
  완전 무료로 이미 로깅되는 값) 두 각도에서 같은 상호작용을 교차검증할 수 있음.
- `select_largest_interaction_fl_algorithm`: Subtask 2.1에서 DoRA와 짝지을
  FL 알고리즘(|interaction|이 더 큰 쪽)을 고름
- `select_best_performing_combination`: Subtask 2.2 local_epochs=5 대상 선정
- `per_category_breakdown` / `client_fairness_variance` / `total_communication_cost`:
  Subtask 2.3 진단용 일반 유틸리티

### 3. 실행 스크립트 (`scripts/run_experiment.py`)
```bash
# Task 1 core, 6회
python scripts/run_experiment.py --peft lora  --fl fedavg
python scripts/run_experiment.py --peft lora  --fl fedprox
python scripts/run_experiment.py --peft lora  --fl scaffold
python scripts/run_experiment.py --peft qlora --fl fedavg
python scripts/run_experiment.py --peft qlora --fl fedprox
python scripts/run_experiment.py --peft qlora --fl scaffold
```

### 4. 결과 분석 스크립트 (`scripts/analyze_interaction.py`)
core 6개 실행 로그를 읽어 상호작용 효과를 출력하고, Task 2 진단 실행에 쓸
`--peft`/`--fl` 인자를 자동으로 알려줌:
```bash
python scripts/analyze_interaction.py
# >>> Subtask 2.1: DoRA를 짝지을 FL 알고리즘 = ...
# >>> Subtask 2.2: local_epochs=5 강건성 점검 대상 = ...
```

### 설계 노트: 평가 비용 관리
`fl_client.py`/`fl_runner.py`(Week 3)에서 이미 PPL은 대표 클라이언트 1개만,
ROUGE-L은 마지막 라운드 1회+카테고리 층화 샘플만 계산하도록 만들어뒀기
때문에, Week 4의 core 6조합 + Task 2 진단 2개(총 8회 실행)를 g5.xlarge
스팟 기준 약 130~175 GPU-hr(로컬 에폭 5배 대상이 LoRA냐 QLoRA냐에 따라
범위가 갈림), 비용으로는 약 $57~77 수준으로 계획하고 있습니다.

---

## 실행 방법 (아직 GPU 환경에서 실행 전 — 문법/배선 검증만 완료)

```bash
pytest tests/ -v   # Week 1~3 테스트 전부, evaluate.py 관련 테스트는 Week 5에 추가 예정
python -c "import scripts.run_experiment, scripts.analyze_interaction"  # import 검증
```

**테스트 결과**: 18 passed, 1 skipped (Week 3와 동일 — `test_evaluate.py`는 Week 5에 추가)

---

## 다음 주 (Week 5) 예고

- `tests/test_evaluate.py`: 상호작용/선정 로직 단위 테스트
- 카테고리별 성능 분해, 수렴 궤적 비교 등 Subtask 2.3 진단 분석 실행

→ `week5` 브랜치 참고.
