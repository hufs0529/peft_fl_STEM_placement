# Dolly-15k FL × PEFT Research Placement — Week 4 진행 보고 (Week 4 Progress Report)

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

**Research Question**: Does the benefit of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist even as PEFT quantization (LoRA→QLoRA) intensifies?

> 이 README는 **Week 1~4 누적** 상태를 반영합니다.

> This README reflects the **cumulative Week 1-4** status.

---

## 진행 상황 요약 (Progress Summary)

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA) | ✅ 완료 |
| Week 2 | Dolly-15k 파이프라인, Dirichlet(α=1) 파티셔닝 | ✅ 완료 |
| Week 3 | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준 | ✅ 완료 |
| **Week 4** | Core 18조합(3압축×2FL×3α) 실행 스크립트 + 압축률×non-IID 상관관계 분석 | ✅ 완료 (이 브랜치) |
| Week 5 | 상관관계 분석 테스트 + 카테고리별 진단 | 예정 (`week5`) |
| Week 6 | 최종 리포트 | 예정 |

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, PEFT wiring (LoRA/QLoRA/DoRA) | ✅ Done |
| Week 2 | Dolly-15k pipeline, Dirichlet(α=1) partitioning | ✅ Done |
| Week 3 | FedProx/SCAFFOLD integration, checkpoint/convergence criteria | ✅ Done |
| **Week 4** | Core 18-combination (3 compression × 2 FL × 3 alpha) run script + compression×non-IID correlation analysis | ✅ Done (this branch) |
| Week 5 | Correlation analysis tests + per-category diagnostics | Planned (`week5`) |
| Week 6 | Final report | Planned |

---

## Week 1~3 요약 (Week 1-3 Summary)
- **Week 1**: PEFT 배선(LoRA/QLoRA/DoRA), 파라미터 헬퍼, wiring 테스트
- **Week 2**: Dolly-15k 파이프라인, Dirichlet(α=1) 파티셔닝
- **Week 3**: FedProx/SCAFFOLD, 체크포인트, 수렴 기준, 수동 FL 라운드 루프,
  FL 통합 테스트(18 passed)

- **Week 1**: PEFT wiring (LoRA/QLoRA/DoRA), parameter helpers, wiring tests
- **Week 2**: Dolly-15k pipeline, Dirichlet(α=1) partitioning
- **Week 3**: FedProx/SCAFFOLD, checkpointing, convergence criteria, manual FL round loop,
  FL integration tests (18 passed)

## Week 4에서 한 일 (What Was Done in Week 4)

> **설계 변경 안내**: 아래는 Week 4 시점에 진행한 작업이며, 이후 Week 5에
> 지도교수 피드백을 받아 실험 설계가 core 6조합(3FL×2PEFT)에서 core
> 18조합(3압축×2FL×3α)으로 개정됐습니다. 이 절은 개정된 설계를 그대로
> 반영합니다 — 개정 배경은 아래 "설계 노트"와 `week5`/`week6` 브랜치를
> 참고하세요.
>
> **Design change notice**: the work below was done as of Week 4; the
> experiment design was later revised in Week 5, per advisor feedback,
> from the core 6 combinations (3 FL × 2 PEFT) to the core 18
> combinations (3 compression × 2 FL × 3 alpha). This section reflects
> the revised design directly — see the "Design Note" below and the
> `week5`/`week6` branches for the background.

### 1. 본실험 설정 (`configs/experiment_config.yaml`) (Main Experiment Configuration)
Qwen2.5-1.5B-Instruct(지도교수 피드백으로 3B에서 축소), 8클라이언트, 최대
10라운드, effective batch 16. `peft.qlora_bits`(4/8)와 `partitioning.alpha`가
`scripts/run_experiment.py`의 `--qlora-bits`/`--alpha`로 override됨.
`rouge_l_sample_size`로 생성 평가 비용 통제.

Qwen2.5-1.5B-Instruct (downsized from 3B per advisor feedback), 8 clients,
up to 10 rounds, effective batch 16. `peft.qlora_bits` (4/8) and
`partitioning.alpha` are overridden via `--qlora-bits`/`--alpha` in
`scripts/run_experiment.py`. `rouge_l_sample_size` controls the cost of
generation evaluation.

### 2. 압축률×non-IID 상관관계 계산 (`src/evaluate.py`) — **이 프로젝트의 핵심 산출물** (Compression×Non-IID Correlation — This Project's Key Deliverable)
- `compute_compression_alpha_trend(run_results, performance_field)`: 각 (FL, α)
  지점에서 "압축 페널티"(QLoRA-4bit 성능 − LoRA(무압축) 성능)를 구하고,
  α가 작아질수록(non-IID가 강해질수록) 이 페널티가 커지는지를 Pearson
  상관계수로 계산 — 지도교수 피드백에 대한 직접적인 정량적 답
- `select_best_performing_combination`: Subtask 2.2 local_epochs=5 대상 선정
  (18조합 전체 대상으로 확장)
- `per_category_breakdown` / `client_fairness_variance` / `total_communication_cost`:
  Subtask 2.3 진단용 일반 유틸리티
- `compute_interaction_effects` / `select_largest_interaction_fl_algorithm`(기존
  core 6조합·SCAFFOLD 대상 상호작용 계산)은 그대로 남아있지만 새 core
  설계에서는 호출하지 않음 — 필요 시 별도 진단으로 재사용 가능

- `compute_compression_alpha_trend(run_results, performance_field)`: at each
  (FL, alpha) point, computes the "compression penalty" (QLoRA-4bit minus
  LoRA/uncompressed performance) and its Pearson correlation with alpha —
  a positive correlation. Whether that penalty grows as alpha decreases
  (non-IID intensifies) is the direct quantitative answer to the
  advisor's question.
- `select_best_performing_combination`: selects the target for Subtask 2.2
  local_epochs=5 (now scoped over all 18 combinations)
- `per_category_breakdown` / `client_fairness_variance` / `total_communication_cost`:
  general-purpose utilities for Subtask 2.3 diagnostics
- `compute_interaction_effects` / `select_largest_interaction_fl_algorithm`
  (the original core-6/SCAFFOLD interaction computation) are kept as-is but
  are no longer called by the new core design — available for a separate
  diagnostic if needed later

### 3. 실행 스크립트 (`scripts/run_experiment.py`) (Run Script)
```bash
# Core, 18회 (3압축 x 2FL x 3alpha)
for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    for alpha in 0.1 1 10; do
      python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
    done
  done
done
```

### 4. 결과 분석 스크립트 (`scripts/analyze_interaction.py`) (Result Analysis Script)
18개 실행 로그를 읽어 압축률×α 격자와 상관계수를 출력하고, Task 2 진단
실행에 쓸 조합을 자동으로 알려줌:
```bash
python scripts/analyze_interaction.py
# === 18조합 격자 (val_perplexity, 낮을수록 좋음) ===
# === 압축률 x Dirichlet alpha 상관관계 (PPL 기준) ===
#   fedavg: ... alpha-페널티 상관계수 = -0.XXXX (음수면 non-IID가 강할수록 압축 페널티가 커짐)
# >>> Subtask 2.2: local_epochs=5 강건성 점검 대상 = ...
```

Reads the logs of the 18 runs, prints the compression×alpha grid and
correlation, and automatically reports the combination to use for the
Task 2 diagnostic run (see the code block above for invocation and
sample output).

### 설계 노트: 지도교수 피드백에 따른 설계 개정 (Design Note: Redesign Following Advisor Feedback)
Week 4 시점에는 원래 core 6조합(3FL×2PEFT) + Task 2 진단 2개(DoRA,
local_epochs=5) 총 8회 실행을 계획했으나, Week 5에 지도교수로부터 "이
연구의 중점을 압축률과 non-IID 강도의 상관관계로 두는 게 좋겠다"는
피드백을 받아 core 설계를 아래와 같이 개정했습니다:

- **압축률 3단계** (LoRA=무압축 / QLoRA 8bit / QLoRA 4bit) × **FL 2종**
  (FedAvg/FedProx, SCAFFOLD는 core에서 제외 — 코드/테스트는 유지) ×
  **Dirichlet α 3단계** (0.1/1/10) = **18회 실행**
- 모델을 Qwen2.5-3B → **Qwen2.5-1.5B-Instruct**로 축소(지도교수 승인) —
  Qwen 계열 유지, GPU 비용 절감
- `fl_client.py`/`fl_runner.py`(Week 3)에서 이미 PPL은 대표 클라이언트
  1개만, ROUGE-L은 마지막 라운드 1회+카테고리 층화 샘플만 계산하도록
  만들어뒀기 때문에, 1.5B 모델 기준 1회당 약 5.5~7.5 GPU-hr로 추정 —
  18회 + Task 2 진단 1개(local_epochs=5) 총 19회 실행을 g5.xlarge 스팟
  기준 약 **105~143 GPU-hr, 비용 약 $35~84**로 계획하고 있습니다.

As of Week 4, the original plan was 8 runs total (the core 6 combinations
[3 FL × 2 PEFT] plus 2 Task 2 diagnostic runs — DoRA and
local_epochs=5). In Week 5, the advisor gave feedback that "this
project's focus should be the correlation between compression rate and
non-IID intensity," so the core design was revised as follows:

- **3 compression levels** (LoRA=uncompressed / QLoRA 8-bit / QLoRA
  4-bit) × **2 FL algorithms** (FedAvg/FedProx — SCAFFOLD is dropped from
  the core design, though its code/tests are kept) × **3 Dirichlet alpha
  levels** (0.1/1/10) = **18 runs**
- The model was downsized from Qwen2.5-3B to **Qwen2.5-1.5B-Instruct**
  (advisor-approved) — staying within the Qwen family while cutting GPU
  cost
- Since `fl_client.py`/`fl_runner.py` (Week 3) already compute PPL over
  only one representative client and ROUGE-L only once at the last round
  over a category-stratified sample, each run is estimated at roughly
  5.5-7.5 GPU-hr on the 1.5B model — we plan for the 18 runs plus 1 Task
  2 diagnostic run (local_epochs=5), 19 runs total, to take roughly
  **105-143 GPU-hr on a g5.xlarge spot instance, costing roughly
  $35-84**.

---

## 실행 방법 (아직 GPU 환경에서 실행 전 — 문법/배선 검증만 완료) (How to Run — Not Yet Executed on a GPU Environment, Only Syntax/Wiring Validated So Far)

```bash
pytest tests/ -v   # Week 1~3 테스트 전부, evaluate.py 관련 테스트는 Week 5에 추가 예정
python -c "import scripts.run_experiment, scripts.analyze_interaction"  # import 검증
```

**테스트 결과**: 18 passed, 2 skipped (QLoRA 4bit/8bit, GPU 필요 — `test_evaluate.py`는 Week 5에 추가)

**Test results**: 18 passed, 2 skipped (QLoRA 4-bit/8-bit, requires GPU —
`test_evaluate.py` will be added in Week 5)

---

## 다음 주 (Week 5) 예고 (Preview of Next Week)

- `tests/test_evaluate.py`: `compute_compression_alpha_trend`(압축률×α 상관관계)
  단위 테스트 + 기존 선정 로직 테스트
- 카테고리별 성능 분해 등 Subtask 2.3 진단 분석 실행

- `tests/test_evaluate.py`: unit tests for `compute_compression_alpha_trend`
  (the compression×alpha correlation) plus the existing selection-logic tests
- Run Subtask 2.3 diagnostic analyses such as per-category performance
  breakdown

→ `week5` 브랜치 참고.

→ See the `week5` branch.
