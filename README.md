# Dolly-15k FL × PEFT Research Placement

6주 연구 실습 프로젝트 코드베이스. 연구계획서 *"Does Drift-Correction
Sophistication Interact with PEFT Quantisation Noise in Federated LLM
Fine-Tuning?"*를 그대로 코드로 구현했습니다.

A 6-week research placement project codebase. It implements the research
proposal *"Does Drift-Correction Sophistication Interact with PEFT
Quantisation Noise in Federated LLM Fine-Tuning?"* directly in code.

---

## 0. 이 프로젝트가 묻는 질문 (연구계획서 Aims, 코드와의 대응) (0. The Question This Project Asks — Research Proposal Aims Mapped to Code)

| Aim | 질문 요약 | 답을 만드는 코드 |
|---|---|---|
| **Aim 1** (Task 1) | 교정 정교함(FedAvg→FedProx)의 이점이 PEFT 압축이 강해져도(무압축→8bit→4bit) 일관되게 유지되는가? 이 이점이 non-IID 강도(Dirichlet α)와 상관관계가 있는가? 3압축×2FL×3α factorial로 검증. | `src/fl_client.py`(FedProx proximal term), `src/evaluate.py::compute_compression_alpha_trend`(압축률×α 상관관계 계산) |
| **Aim 2** (Task 2) | 발견된 상관관계 하에서, 성능이 가장 좋았던 조합이 local_epochs를 늘려도 견고한가? | `src/evaluate.py::select_best_performing_combination`, `src/evaluate.py::per_category_breakdown` |

| Aim | Question summary | Code that produces the answer |
|---|---|---|
| **Aim 1** (Task 1) | Does the benefit of drift-correction sophistication (FedAvg→FedProx) hold consistently as PEFT compression intensifies (uncompressed→8-bit→4-bit)? Is this benefit correlated with non-IID intensity (Dirichlet alpha)? Verified with a 3 compression × 2 FL × 3 alpha factorial design. | `src/fl_client.py` (FedProx proximal term), `src/evaluate.py::compute_compression_alpha_trend` (compression×alpha correlation) |
| **Aim 2** (Task 2) | Given the observed correlation, is the best-performing combination robust to a larger local_epochs? | `src/evaluate.py::select_best_performing_combination`, `src/evaluate.py::per_category_breakdown` |

> **설계 변경 안내 (Week 5, 지도교수 피드백)**: 이 프로젝트는 원래
> 3FL(FedAvg/FedProx/SCAFFOLD)×2PEFT(LoRA/QLoRA)의 6조합으로 "교정
> 정교함 × 압축 강도"를 검증할 계획이었습니다. Week 5에 지도교수로부터
> "연구의 중점을 압축률과 non-IID 강도의 상관관계로 두는 게 좋겠다"는
> 피드백을 받아, SCAFFOLD를 core에서 빼고(코드/테스트는 유지) 압축
> 강도를 3단계(무압축/8bit/4bit)로, Dirichlet α를 3단계(0.1/0.5/1.0)로
> 넓힌 **3×2×3=18조합** 설계로 개정했습니다. 원래 연구질문·6조합 설계는
> `week4` 브랜치 이전 시점의 히스토리로 남아있습니다.
>
> **Design change notice (Week 5, advisor feedback)**: this project
> originally planned to verify "correction sophistication × compression
> intensity" with 6 combinations of 3 FL algorithms
> (FedAvg/FedProx/SCAFFOLD) × 2 PEFT methods (LoRA/QLoRA). In Week 5, the
> advisor gave feedback that "the project's focus should be the
> correlation between compression rate and non-IID intensity," so the
> design was revised to drop SCAFFOLD from the core (its code/tests are
> kept) and widen compression to 3 levels (uncompressed/8-bit/4-bit) and
> Dirichlet alpha to 3 levels (0.1/0.5/1.0) — a **3×2×3 = 18-combination**
> design. The original research question and 6-combination design remain
> in the history prior to the `week4` branch.

**핵심 설계 원칙**: 이 코드베이스는 18개 조합(3압축×2FL×3α)의 core 실험으로
"압축 강도 × non-IID 강도"의 격자를 직접 채우고, Task 2는 이 격자가
보여준 결과를 **왜** 나타나는지 설명하는 데 필요한 최소한의 진단
실험(local-epoch 강건성 점검) 하나로만 한정합니다 — 관련 없는 축(Prompt
Tuning, 부분 참여, LLM-judge, 중앙집중형 베이스라인)은 이번 연구계획서
범위 밖이므로 포함하지 않습니다.

**Core design principle**: This codebase fills the "compression intensity ×
non-IID intensity" grid directly with 18 core combinations (3 compression ×
2 FL × 3 alpha), and Task 2 is limited to exactly the one diagnostic
experiment (local-epoch robustness check) needed to explain **why** the
result shown by that grid appears — unrelated axes (Prompt Tuning, partial
participation, an LLM-judge, centralised baselines) are out of scope for
this research proposal and are not included.

---

## 1. 폴더 구조 (1. Folder Structure)

```
dolly15k-fl-peft-v2/
├── README.md                     # 이 파일
├── requirements.txt
├── configs/
│   ├── dev_config.yaml           # Week 1~3 검증용 (gpt2, 2클라이언트, 2라운드)
│   └── experiment_config.yaml    # Week 4 본실험용 (Qwen2.5-1.5B, 8클라이언트, 10라운드; Week 5 개정)
├── src/                          # dev/experiment 공용 핵심 로직
│   ├── models.py                   # LoRA/QLoRA(4bit·8bit, core), DoRA(미사용 진단 대조군) 래핑
│   ├── data.py                     # Dolly-15k 로딩, 토크나이징, held-out 스플릿
│   ├── partitioning.py             # Dirichlet(alpha=0.5) 비IID + 카테고리 리샘플링
│   ├── communication.py            # FL 파라미터 왕복 + payload 크기 계산
│   ├── fl_client.py                # 공통 클라이언트 (FedProx proximal term, warmup, VRAM/latency, ROUGE-L)
│   ├── fl_runner.py                # 수동 라운드 루프 — 수렴기준/체크포인트/로깅/W&B 전부 배선
│   ├── scaffold.py                 # SCAFFOLD 실제 구현 (control variate) — core에서는 제외, 코드/테스트만 유지
│   ├── convergence.py              # 3라운드 연속 <1% 개선 시 조기종료
│   ├── checkpointing.py            # 라운드별 저장/재개
│   ├── metrics.py                  # VRAM, latency, ROUGE-L, trainable param count
│   └── evaluate.py                 # 카테고리별 분해, client fairness, 압축률x α 상관관계 계산
├── tests/                        # 아래 "테스트 구성" 절 참고
├── scripts/
│   ├── run_dev_pilot.py            # Week 1: 실제 모델 단일 클라이언트 검증
│   ├── run_experiment.py           # Task 1 core 18조합(3압축x2FLx3α) + Task 2 진단 실행 진입점
│   └── analyze_interaction.py      # Subtask 1.3/2.2: 압축률x α 상관관계 분석 + 진단 대상 선정
└── results/
    ├── checkpoints/{run_name}/round_NNN.pt   # 라운드별 체크포인트 (git 추적 제외)
    ├── logs/{run_name}_rounds.jsonl          # 라운드별 전체 지표
    ├── logs/{run_name}_generations.jsonl     # 마지막 생성 라운드의 예측/참조 (Subtask 2.3 입력)
    └── figures/                              # Week 5 시각화 출력
```

```
dolly15k-fl-peft-v2/
├── README.md                     # this file
├── requirements.txt
├── configs/
│   ├── dev_config.yaml           # for Week 1-3 verification (gpt2, 2 clients, 2 rounds)
│   └── experiment_config.yaml    # for the Week 4 main experiment (Qwen2.5-1.5B, 8 clients, 10 rounds; revised in Week 5)
├── src/                          # core logic shared by dev/experiment
│   ├── models.py                   # wraps LoRA/QLoRA (4-bit/8-bit, core) and DoRA (unused diagnostic control)
│   ├── data.py                     # Dolly-15k loading, tokenisation, held-out split
│   ├── partitioning.py             # Dirichlet (alpha=0.5) non-IID + category resampling
│   ├── communication.py            # FL parameter round-trip + payload size computation
│   ├── fl_client.py                # shared client (FedProx proximal term, warmup, VRAM/latency, ROUGE-L)
│   ├── fl_runner.py                # manual round loop — wires up convergence, checkpointing, logging, W&B
│   ├── scaffold.py                 # actual SCAFFOLD implementation (control variate) — dropped from core, code/tests kept
│   ├── convergence.py              # early stop on 3 consecutive rounds of <1% improvement
│   ├── checkpointing.py            # per-round save/resume
│   ├── metrics.py                  # VRAM, latency, ROUGE-L, trainable param count
│   └── evaluate.py                 # per-category breakdown, client fairness, compression x alpha correlation
├── tests/                        # see the "Test Composition" section below
├── scripts/
│   ├── run_dev_pilot.py            # Week 1: single-client verification with the real model
│   ├── run_experiment.py           # entry point for Task 1's core 18 combinations (3 compression x 2 FL x 3 alpha) + Task 2 diagnostics
│   └── analyze_interaction.py      # Subtask 1.3/2.2: compression x alpha correlation analysis + diagnostic target selection
└── results/
    ├── checkpoints/{run_name}/round_NNN.pt   # per-round checkpoints (excluded from git tracking)
    ├── logs/{run_name}_rounds.jsonl          # full per-round metrics
    ├── logs/{run_name}_generations.jsonl     # predictions/references from the final generation round (Subtask 2.3 input)
    └── figures/                              # Week 5 visualisation outputs
```

---

## 2. Task 1 core 설계가 코드에 어떻게 반영되는가 (2. How the Task 1 Core Design Is Reflected in the Code)

개정된 core 설계(**3압축 × 2FL × 3α = 18조합**)는 하나의 스크립트
(`run_experiment.py`)에 `--peft`(+`--qlora-bits`), `--fl`, `--alpha` 세
축의 인자로 압축되어 있습니다. 즉 아래 18개 명령이 core 전체입니다:

The revised core design (**3 compression × 2 FL × 3 alpha = 18
combinations**) is compressed into a single script (`run_experiment.py`)
with three axes of arguments: `--peft` (+ `--qlora-bits`), `--fl`, and
`--alpha`. In other words, the 18 commands below make up the entire core:

```bash
for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    for alpha in 0.1 0.5 1.0; do
      python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
    done
  done
done
```

18개가 모두 끝나면:

Once all 18 have finished:

```bash
python scripts/analyze_interaction.py
```

를 실행합니다. 이 스크립트는 세 가지를 계산합니다:

is run. This script computes three things:

1. **압축률×α 격자** — 18개 조합의 val_perplexity를 (압축, FL, α) 격자로
   출력합니다.
2. **Subtask 1.3 압축률×non-IID 상관관계** — 각 (FL, α) 지점에서
   "압축 페널티"(QLoRA-4bit 성능 − LoRA(무압축) 성능)를 구하고, α가
   작아질수록(non-IID가 강해질수록) 이 페널티가 커지는지를 Pearson
   상관계수로 계산합니다. 이것이 지도교수 피드백("압축률과 non-IID
   강도의 상관관계를 보라")에 대한 직접적인 정량적 답입니다
   (`src/evaluate.py::compute_compression_alpha_trend`).
3. **Subtask 2.2 대상 선정** — 18조합 중 task performance가 가장 좋았던
   조합을 골라, local_epochs=5 강건성 점검 대상으로 삼습니다.

1. **Compression×alpha grid** — prints the val_perplexity of all 18
   combinations as a (compression, FL, alpha) grid.
2. **Subtask 1.3 compression×non-IID correlation** — at each (FL, alpha)
   point, computes the "compression penalty" (QLoRA-4bit minus
   LoRA/uncompressed performance), and its Pearson correlation with
   alpha to check whether the penalty grows as alpha decreases (non-IID
   intensifies). This is the direct quantitative answer to the advisor's
   feedback ("look at the correlation between compression rate and
   non-IID intensity") (`src/evaluate.py::compute_compression_alpha_trend`).
3. **Subtask 2.2 target selection** — picks the combination among the 18
   with the best task performance as the target for the local_epochs=5
   robustness check.

### Task 2 진단 실행 (Task 2 Diagnostic Execution)

| 무엇을 검증하는가 | 명령 |
|---|---|
| Local epoch 1 vs 5 강건성 점검 (best-performing 조합에 적용) | `run_experiment.py --peft <best_peft> --fl <best_fl> --alpha <best_alpha> [--qlora-bits 8] --local-epochs 5` |

| What it verifies | Command |
|---|---|
| Local epoch 1 vs 5 robustness check (applied to the best-performing combination) | `run_experiment.py --peft <best_peft> --fl <best_fl> --alpha <best_alpha> [--qlora-bits 8] --local-epochs 5` |

`<best_peft>`/`<best_fl>`/`<best_alpha>`는 `analyze_interaction.py`
출력에서 그대로 복사해 쓰면 됩니다.

`<best_peft>` / `<best_fl>` / `<best_alpha>` can simply be copied from the
output of `analyze_interaction.py`.

Subtask 2.3(카테고리별 분해)은 추가 GPU 비용 없이 `src/evaluate.py`의
`per_category_breakdown`과 `results/logs/*_rounds.jsonl`에 이미 기록된
라운드별 지표에서 바로 계산합니다.

Subtask 2.3 (per-category breakdown) is computed directly from
`per_category_breakdown` in `src/evaluate.py` and the per-round metrics
already logged in `results/logs/*_rounds.jsonl`, with no additional GPU
cost.

> SCAFFOLD/DoRA 관련 코드(`src/scaffold.py`, `--peft dora`,
> `--fl scaffold`)는 여전히 동작하고 테스트도 통과하지만, 위 core
> 18조합에는 포함되지 않습니다. 필요하면 별도 진단으로 수동 실행할 수
> 있습니다.
>
> The SCAFFOLD/DoRA code (`src/scaffold.py`, `--peft dora`,
> `--fl scaffold`) still works and its tests still pass, but it is not
> included in the 18 core combinations above. It can still be run
> manually as a separate diagnostic if needed.

---

## 3. 각 모듈이 연구계획서의 어느 부분과 대응하는가 (3. How Each Module Maps to the Research Proposal)

### `src/models.py` — Subtask 1.1, 2.1
- **core**: LoRA(r=8, α=16, dropout=0.05, 무압축), QLoRA(`qlora_bits`로
  4bit NF4 또는 8bit INT8 선택 — 압축률 스윕용)
- **미사용 진단**: DoRA(`use_dora=True`, 양자화 없음)는 여전히 코드에
  있지만, 개정된 core 18조합 설계에는 포함되지 않습니다.
- 다른 PEFT 방식(Prompt Tuning, Adapter Tuning 등)은 이번 연구계획서의
  범위 밖이므로 포함하지 않았습니다 — `lora`/`qlora`/`dora` 이외의 값을
  넘기면 `ValueError`가 나는 것도 의도된 동작이며,
  `tests/test_model_wiring.py`에서 이를 확인합니다.

### `src/models.py` — Subtask 1.1, 2.1
- **core**: LoRA (r=8, α=16, dropout=0.05, uncompressed), QLoRA
  (`qlora_bits` selects 4-bit NF4 or 8-bit INT8 — for the
  compression-rate sweep)
- **unused diagnostic**: DoRA (`use_dora=True`, no quantisation) is still
  in the code, but is not part of the revised core 18-combination design.
- Other PEFT methods (Prompt Tuning, Adapter Tuning, etc.) are out of
  scope for this research proposal and are not included — passing any
  value other than `lora`/`qlora`/`dora` intentionally raises a
  `ValueError`, which `tests/test_model_wiring.py` verifies.

### `src/scaffold.py` — core에서 제외됨(코드/테스트는 유지) (Dropped from Core — Code/Tests Kept)
FedAvg/FedProx는 Flower의 표준 weighted-average 집계로 충분하지만, SCAFFOLD는
클라이언트별 control variate를 별도로 유지·통신해야 해서 `fl_runner.py`가 이
파일의 `scaffold_client_fit()`/`scaffold_aggregate()`를 직접 호출하는 별도
경로로 처리합니다. **간소화 구현(Option II 근사)**입니다. Week 5 지도교수
피드백으로 core 18조합 설계에서는 제외됐지만, `--fl scaffold`로 여전히
수동 실행 가능하고 `tests/test_fl_integration.py`도 계속 통과합니다.

FedAvg/FedProx are adequately handled by Flower's standard
weighted-average aggregation, but SCAFFOLD needs to separately maintain
and communicate a per-client control variate, so `fl_runner.py` routes it
through a separate path that directly calls this file's
`scaffold_client_fit()` / `scaffold_aggregate()`. It is a **simplified
implementation (Option II approximation)**. It was dropped from the core
18-combination design per Week 5 advisor feedback, but can still be run
manually via `--fl scaffold`, and `tests/test_fl_integration.py` continues
to pass.

### `src/fl_client.py` — Subtask 1.1, 1.2
- FedProx일 때만 `(mu/2)*||local-global||^2` proximal term이 loss에 추가됩니다
  (`is_fedprox` 분기, μ=0.01).
- 첫 라운드(`server_round==1`)에만 linear warmup이 적용됩니다.
- `track_vram_and_latency()`로 감싸서 Peak VRAM/Training Latency를 자동 측정합니다.
- `evaluate()`는 매 라운드 PPL(teacher-forcing 순전파)만 계산하고, ROUGE-L
  (자기회귀 생성이 필요해 훨씬 비쌈)은 `run_generation_metrics=True`일
  때만 별도의 `rouge_eval_dataset`(카테고리 층화 샘플)에 대해 계산합니다
  — 언제 이 플래그가 켜지는지는 `fl_runner.py`가 결정합니다(아래).

### `src/fl_client.py` — Subtask 1.1, 1.2
- The `(mu/2)*||local-global||^2` proximal term is added to the loss only
  under FedProx (`is_fedprox` branch, μ=0.01).
- Linear warmup is applied only in the first round (`server_round==1`).
- Wrapped in `track_vram_and_latency()` to automatically measure Peak
  VRAM/Training Latency.
- `evaluate()` computes only PPL (a teacher-forcing forward pass) every
  round; ROUGE-L (which needs much more expensive autoregressive
  generation) is computed only when `run_generation_metrics=True`, against
  a separate `rouge_eval_dataset` (a category-stratified sample) — when
  this flag gets turned on is decided by `fl_runner.py` (below).

### `src/fl_runner.py` — Subtask 1.2, 2.3
- **수렴 기준**: `ConvergenceTracker`가 매 라운드 검증 loss를 받아 3라운드 연속
  1% 미만 개선이면 루프를 멈춥니다(≤10라운드).
- **체크포인트**: 매 라운드 저장, 실행 시작 시 자동으로 마지막 체크포인트에서
  재개(`resume_or_start_fresh`) — 세션이 끊겨도 안전합니다.
- **평가 비용 통제**: 매 라운드 집계 직후 모든 클라이언트는 동일한 global
  파라미터로 덮어써지고 동일한 공용 held-out set을 보므로, 대표 클라이언트
  (`clients[0]`) 1개만 평가합니다 — 8명을 전부 평가해도 결과가 같아
  나머지는 순수 중복 계산이기 때문입니다. ROUGE-L(생성 필요)은 매 라운드가
  아니라 **루프 종료 후 최종 global 파라미터로 딱 1회만**, 카테고리 층화
  샘플(`config['data']['rouge_l_sample_size']`)에 대해 계산합니다.
- **로깅**: `communication_bytes_this_round`(통신비용의 재료)를 매 라운드
  `results/logs/{run_name}_rounds.jsonl`에 씁니다.
- **W&B**: `config['logging']['use_wandb']=true`면 실시간으로 같은 지표를
  W&B 대시보드에도 올립니다.

### `src/fl_runner.py` — Subtask 1.2, 2.3
- **Convergence criterion**: `ConvergenceTracker` takes the validation loss
  every round and stops the loop after 3 consecutive rounds of <1%
  improvement (≤10 rounds).
- **Checkpointing**: saved every round, and automatically resumed from the
  last checkpoint at start-up (`resume_or_start_fresh`) — safe even if a
  session is interrupted.
- **Evaluation cost control**: right after aggregation, every client is
  overwritten with the same global parameters and sees the same shared
  held-out set, so only the representative client (`clients[0]`) is
  evaluated — evaluating all 8 would give the same result, so the rest
  would be pure duplicate computation. ROUGE-L (which needs generation) is
  computed **only once, after the loop ends, on the final global
  parameters**, against a category-stratified sample
  (`config['data']['rouge_l_sample_size']`), rather than every round.
- **Logging**: writes `communication_bytes_this_round` (the raw material
  for communication cost) to `results/logs/{run_name}_rounds.jsonl` every
  round.
- **W&B**: if `config['logging']['use_wandb']=true`, the same metrics are
  also pushed to the W&B dashboard in real time.

### `src/evaluate.py` — Subtask 1.3, 2.2, 2.3
- `compute_compression_alpha_trend`: **지도교수 피드백에 대한 직접적인
  답**을 계산하는 함수. 18조합의 성능값에서 각 (FL, α) 지점의 "압축
  페널티"(QLoRA-4bit − LoRA)를 구하고, α와 페널티의 Pearson 상관계수를
  산출합니다. `performance_field`로 `val_perplexity`(성능)와
  `rounds_run`(수렴 속도, 완전 무료로 이미 로깅된 값) 둘 다에 대해
  계산해 두 지표가 같은 방향을 가리키는지 교차검증합니다
  (`scripts/analyze_interaction.py`). 페널티가 alpha와 무관(분산 0)하면
  NaN 대신 0.0을 반환하도록 처리돼 있습니다.
- `select_best_performing_combination`: Subtask 2.2에서 local_epochs=5
  점검 대상을 고릅니다(18조합 전체 대상).
- `per_category_breakdown`, `total_communication_cost`: Subtask 2.3
  분석에서 그대로 불러 쓰는 함수들. 추가 GPU 비용 없이 이미 로깅된
  데이터에서 계산됩니다.
- `client_fairness_variance`: 순수 유틸리티 함수로만 남아있습니다 —
  현재 held-out set이 클라이언트 간 전역 공유라 파이프라인에서는 호출하지
  않습니다(항상 동일값이라 무의미). 아래 "알려진 한계" 참고.
- `compute_interaction_effects` / `select_largest_interaction_fl_algorithm`:
  원래 core 6조합·SCAFFOLD 대상 상호작용 계산 함수. 코드와 테스트는
  그대로 남아있지만 새 core 설계(`analyze_interaction.py`)에서는
  호출하지 않습니다.

### `src/evaluate.py` — Subtask 1.3, 2.2, 2.3
- `compute_compression_alpha_trend`: the function that computes **the
  direct answer to the advisor's feedback**. From the performance values
  of the 18 combinations, it derives the "compression penalty"
  (QLoRA-4bit minus LoRA) at each (FL, alpha) point, and its Pearson
  correlation with alpha. It is computed for both `val_perplexity`
  (performance) and `rounds_run` (convergence speed, a value already
  logged for free) via `performance_field`, so the two metrics can be
  cross-checked for whether they point the same direction
  (`scripts/analyze_interaction.py`). Returns 0.0 instead of NaN when the
  penalty is independent of alpha (zero variance).
- `select_best_performing_combination`: picks the target for the
  local_epochs=5 check in Subtask 2.2 (now scoped over all 18
  combinations).
- `per_category_breakdown`, `total_communication_cost`: general utilities
  used as-is in Subtask 2.3 analysis. Computed from already-logged data at
  no additional GPU cost.
- `client_fairness_variance`: remains a pure utility function only — since
  the current held-out set is shared globally across clients, the
  pipeline never calls it (it would always be the same value, so it's
  meaningless). See "Known Limitations" below.
- `compute_interaction_effects` / `select_largest_interaction_fl_algorithm`:
  the original core-6/SCAFFOLD interaction computation functions. Their
  code and tests remain, but they are no longer called by the new core
  design (`analyze_interaction.py`).

---

## 4. 워크플로우 (6주 타임라인과 매핑) (4. Workflow — Mapped to the 6-Week Timeline)

| 주차 | 실행 명령 | 검증 대상 |
|---|---|---|
| Week 1 | `pytest tests/test_model_wiring.py` → `python scripts/run_dev_pilot.py --stage single_client` | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA), 단일 클라이언트 loss 감소 |
| Week 2 | `pytest tests/test_roundtrip.py tests/test_partitioning.py` | Dolly-15k 파이프라인, 파라미터 왕복, Dirichlet(α=0.5) 파티셔닝 검증 |
| Week 3 | `pytest tests/test_fl_integration.py tests/test_convergence.py tests/test_checkpointing.py` | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준, 축소 규모 6조합 예행연습 |
| Week 4 | `run_experiment.py` × 18(core) → `analyze_interaction.py` → local-epoch 강건성 점검 | Subtask 1.2 본 실험(개정 설계) + Subtask 2.2 진단 |
| Week 5 | `pytest tests/test_evaluate.py` → 분석 노트북에서 `src/evaluate.py` 함수 직접 호출 | Subtask 1.3 압축률×non-IID 상관관계 분석, Subtask 2.3 카테고리별 진단 |
| Week 6 | 리포트 작성 (설계 개정 배경·SCAFFOLD core 제외·single-seed 한계 명시) + 발표 | — |

| Week | Command run | What it verifies |
|---|---|---|
| Week 1 | `pytest tests/test_model_wiring.py` → `python scripts/run_dev_pilot.py --stage single_client` | model+GPU environment check, PEFT wiring (LoRA/QLoRA/DoRA), single-client loss decrease |
| Week 2 | `pytest tests/test_roundtrip.py tests/test_partitioning.py` | Dolly-15k pipeline, parameter round-trip, Dirichlet (α=0.5) partitioning verification |
| Week 3 | `pytest tests/test_fl_integration.py tests/test_convergence.py tests/test_checkpointing.py` | FedProx/SCAFFOLD integration, checkpointing/convergence criterion, small-scale rehearsal of the 6 combinations |
| Week 4 | `run_experiment.py` × 18 (core) → `analyze_interaction.py` → local-epoch robustness check | Subtask 1.2 main experiment (revised design) + Subtask 2.2 diagnostic |
| Week 5 | `pytest tests/test_evaluate.py` → calling `src/evaluate.py` functions directly from an analysis notebook | Subtask 1.3 compression×non-IID correlation analysis, Subtask 2.3 per-category diagnostics |
| Week 6 | write the report (noting the redesign background, SCAFFOLD dropped from core, and single-seed limitations) + present | — |

**전체 테스트를 한 번에 돌리려면**:
**To run the entire test suite at once**:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## 5. 테스트 구성 (5. Test Composition)

| 파일 | 무엇을 검증하는가 | 대응 단계 |
|---|---|---|
| `test_model_wiring.py` | LoRA/QLoRA(4bit·8bit)/DoRA 배선, 정의되지 않은 PEFT type이 `ValueError`를 내는지 | Week 1, 2(8bit 추가) |
| `test_roundtrip.py` | FL 파라미터 송수신 시 값/순서 보존 | Week 2 |
| `test_partitioning.py` | 최소 카테고리 임계값 리샘플링, α가 작을수록 실제로 더 비IID한지 | Week 2 |
| `test_fl_integration.py` | FedAvg/FedProx/SCAFFOLD 세 경로 모두 전체 루프 통과, SCAFFOLD의 `local_control` 라운드 간 유지, FedProx fit() 정상 동작 | Week 3 |
| `test_convergence.py` | 3라운드 연속 <1% 개선 시 실제로 멈추는지, 큰 폭 개선 중에는 안 멈추는지 | Week 3 |
| `test_checkpointing.py` | 저장 후 재개 시 라운드/상태가 정확히 복원되는지 | Week 3 |
| `test_evaluate.py` | 카테고리별 분해, fairness variance, **압축률×α 상관관계 계산**(페널티 값·음의 상관관계·상수 페널티일 때 0.0), **최고 성능 조합 선정**, (레거시) 상호작용 효과 계산 | Week 4~5 |

| File | What it verifies | Corresponding week |
|---|---|---|
| `test_model_wiring.py` | LoRA/QLoRA (4-bit/8-bit)/DoRA wiring; that an undefined PEFT type raises `ValueError` | Week 1, 2 (8-bit added) |
| `test_roundtrip.py` | value/order preservation when FL parameters are sent and received | Week 2 |
| `test_partitioning.py` | minimum-category-threshold resampling; that a smaller α actually produces a more non-IID split | Week 2 |
| `test_fl_integration.py` | all three of FedAvg/FedProx/SCAFFOLD complete the full loop; SCAFFOLD's `local_control` is preserved across rounds; FedProx's `fit()` runs correctly | Week 3 |
| `test_convergence.py` | actually stops after 3 consecutive rounds of <1% improvement; does not stop while improvements are large | Week 3 |
| `test_checkpointing.py` | round/state are restored exactly after save-then-resume | Week 3 |
| `test_evaluate.py` | per-category breakdown, fairness variance, **compression×alpha correlation** (penalty values, negative correlation, 0.0 for constant penalty), **selecting the best-performing combination**, (legacy) interaction-effect computation | Week 4-5 |

---

## 6. 실행 환경 (6. Execution Environment)

- **Week 1~3 (개발/디버깅)**: Google Colab Pro — compute unit 소모가 적은
  미니어처 검증 위주라 기본 구독만으로 충분합니다.
- **Week 4 (본 실험)**: 예약형/스팟 단일 GPU 인스턴스(AWS g5.xlarge 또는
  Lambda Labs A100) — `src/checkpointing.py`가 매 라운드 저장하므로 스팟
  인스턴스가 회수되어도 안전하게 재개할 수 있습니다. (Week 5 갱신: 모델을
  Qwen2.5-1.5B로 축소하면서 g5.xlarge 스팟 기준 core 18회+진단 1회
  총 19회 실행이 약 105~143 GPU-hr, 비용 약 $35~84로 추정됩니다 —
  `week4` 브랜치 설계 노트 참고.)

- **Week 1-3 (development/debugging)**: Google Colab Pro — the base
  subscription is enough since the work is mostly miniature verification
  with low compute-unit consumption.
- **Week 4 (main experiment)**: an on-demand or spot single-GPU instance
  (AWS g5.xlarge or Lambda Labs A100) — since `src/checkpointing.py` saves
  every round, it is safe to resume even if the spot instance is
  reclaimed. (Week 5 update: after downsizing the model to Qwen2.5-1.5B,
  the core 18 runs plus 1 diagnostic run — 19 total — are estimated at
  roughly 105-143 GPU-hr on a g5.xlarge spot instance, costing roughly
  $35-84 — see the design note on the `week4` branch.)

```bash
pip install -r requirements.txt
# bitsandbytes는 GPU 환경에서만 정상 설치/동작합니다.
# bitsandbytes only installs/works correctly in a GPU environment.
```

---

## 7. 알려진 한계 (Week 6 리포트 Limitations에 반영할 것) (7. Known Limitations — To Be Reflected in the Week 6 Report's Limitations Section)

- **SCAFFOLD는 core 18조합에서 제외**됐습니다(Week 5 지도교수 피드백) —
  코드(`src/scaffold.py`)와 테스트는 남아있지만, 본 실험은 FedAvg/FedProx
  둘만 비교합니다. SCAFFOLD처럼 더 정교한 교정 알고리즘에서도 압축률×
  non-IID 상관관계가 동일하게 나타나는지는 이번 스코프에서 답하지
  못합니다. 또한 SCAFFOLD 자체는 여전히 **Option II의 간소화 근사
  구현**입니다.
- 모델을 **Qwen2.5-3B → Qwen2.5-1.5B-Instruct로 축소**했습니다(지도교수
  승인, GPU 비용 절감 목적) — 더 큰 모델에서도 같은 상관관계 패턴이
  유지되는지는 검증하지 못했습니다.
- 각 조합은 **1회씩만 실행**됩니다(반복/시드 없음) — 시간 제약상 불가피한 한계.
- Held-out 평가셋은 **전역 공유 방식**이라 클라이언트별 분포 편향을 완전히
  반영하지 못합니다. 이 때문에 클라이언트별 fairness variance는 (모든
  클라이언트가 매 라운드 동일한 값을 갖게 되어) 의미 있는 신호가 되지
  않아 로깅하지 않습니다 — `src/evaluate.py::client_fairness_variance`는
  순수 유틸리티 함수로만 남아있습니다.
- ROUGE-L은 held-out 전체가 아니라 **카테고리 층화 샘플
  (`data.rouge_l_sample_size`, 기본 200개)**, **마지막 라운드 1회**만
  계산합니다 — 자기회귀 생성 비용이 커서(1,501개 전체를 매 라운드 생성하면
  수십 GPU시간) 통계적으로 안정적인 수준에서 샘플링합니다.
- 압축률×α 상관관계(`compute_compression_alpha_trend`)는 α **3개 지점
  (0.1/0.5/1.0)**만으로 계산한 Pearson 상관계수입니다 — 지점이 적어
  비선형적인 패턴(예: 중간 α에서 페널티가 가장 큰 U자형)은 놓칠 수
  있습니다.
- 코드 전체는 **문법 검증만 마쳤고 실제 GPU 환경에서 아직 실행되지 않았습니다.**
  Week 1 첫날 `pytest tests/`부터 돌려서 실제 동작을 확인하세요.

- **SCAFFOLD is dropped from the core 18 combinations** (Week 5 advisor
  feedback) — its code (`src/scaffold.py`) and tests remain, but the main
  experiment compares only FedAvg and FedProx. Whether the same
  compression×non-IID correlation holds under a more sophisticated
  correction algorithm like SCAFFOLD is not answered within this scope.
  SCAFFOLD itself also remains a **simplified Option II approximation**.
- The model was **downsized from Qwen2.5-3B to Qwen2.5-1.5B-Instruct**
  (advisor-approved, to cut GPU cost) — whether the same correlation
  pattern holds on a larger model was not verified.
- Each combination is **run only once** (no repeats/seeds) — an
  unavoidable limitation given the time constraints.
- The held-out evaluation set is **shared globally**, so it does not fully
  reflect per-client distribution bias. Because of this, per-client
  fairness variance is not logged (every client would have the same value
  every round, so it carries no meaningful signal) —
  `src/evaluate.py::client_fairness_variance` remains a pure utility
  function only.
- ROUGE-L is computed not on the full held-out set but on a
  **category-stratified sample** (`data.rouge_l_sample_size`, 200 by
  default), and **only once, on the final round** — since autoregressive
  generation is expensive (generating all 1,501 examples every round
  would take dozens of GPU-hours), sampling is used at a statistically
  stable level instead.
- The compression×alpha correlation (`compute_compression_alpha_trend`) is
  a Pearson correlation computed from only **3 alpha points (0.1/0.5/1.0)**
  — with so few points, a non-linear pattern (e.g., a U-shape where the
  penalty peaks at a middle alpha) could be missed.
- The entire codebase has **only passed syntax verification and has not
  yet been run in an actual GPU environment.** Start by running
  `pytest tests/` on Week 1's first day to confirm real behaviour.
