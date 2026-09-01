# Dolly-15k FL × PEFT Research Placement — Week 3 진행 보고 (Week 3 Progress Report)

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

**Research Question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist as PEFT quantization intensifies (LoRA→QLoRA)?

> 이 README는 **Week 1~3 누적** 상태를 반영합니다.

> This README reflects the **cumulative state of Weeks 1–3**.

---

## 진행 상황 요약 (Progress Summary)

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA) | ✅ 완료 |
| Week 2 | Dolly-15k 파이프라인, 파라미터 왕복, Dirichlet(α=1) 파티셔닝 | ✅ 완료 |
| **Week 3** | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준, FL 통합 테스트 | ✅ 완료 (이 브랜치) |
| Week 4 | Core 실험 실행(3압축×2FL×3α=18조합) + local-epoch 진단 | 예정 (`week4`) |
| Week 5 | 압축률×non-IID 상관관계 분석 + 카테고리별 진단 | 예정 |
| Week 6 | 최종 리포트 | 예정 |

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model + GPU environment, wire up PEFT (LoRA/QLoRA/DoRA) | ✅ Done |
| Week 2 | Dolly-15k pipeline, parameter round-trip, Dirichlet (α=1) partitioning | ✅ Done |
| **Week 3** | FedProx/SCAFFOLD integration, checkpoint/convergence criteria, FL integration tests | ✅ Done (this branch) |
| Week 4 | Run the core experiment (3 compression × 2 FL × 3 alpha = 18 combinations) + local-epoch diagnostic | Planned (`week4`) |
| Week 5 | Compression×non-IID correlation analysis + per-category diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Week 1~2 요약 (Week 1–2 Summary)

- **Week 1**: `src/models.py`(LoRA/QLoRA/DoRA 배선), `src/communication.py`(파라미터 헬퍼),
  wiring 테스트, dev pilot 스크립트
- **Week 2**: `src/data.py`(Dolly-15k 로딩/토큰화/held-out), `src/partitioning.py`
  (Dirichlet α=1 비IID 분배), roundtrip/partitioning 테스트

- **Week 1**: `src/models.py` (LoRA/QLoRA/DoRA wiring), `src/communication.py`
  (parameter helpers), wiring tests, dev pilot script
- **Week 2**: `src/data.py` (Dolly-15k loading/tokenization/held-out),
  `src/partitioning.py` (Dirichlet α=1 non-IID split), roundtrip/partitioning
  tests

## Week 3에서 한 일 — 이번 주가 가장 큰 덩어리입니다 (What Was Done in Week 3 — This Is the Biggest Chunk So Far)

### 1. 학습/평가 지표 유틸 (`src/metrics.py`)
`track_vram_and_latency()`(학습 루프를 감싸는 context manager로 Peak VRAM·
Training Latency 측정), `compute_rouge_l()`, `generate_responses()`,
`count_trainable_parameters()` — 이후 클라이언트 구현이 바로 가져다 씁니다.

### 1. Training/evaluation metric utilities (`src/metrics.py`)
`track_vram_and_latency()` (a context manager wrapping the training loop to
measure Peak VRAM and Training Latency), `compute_rouge_l()`,
`generate_responses()`, `count_trainable_parameters()` — used directly by the
client implementation that follows.

### 2. 수렴 기준 (`src/convergence.py`)
`ConvergenceTracker`: 최근 3라운드 연속 개선률이 1% 미만이면 수렴 판정.
FL 루프에 배선하기 전에 로직만 독립적으로 먼저 검증했습니다.

### 2. Convergence criterion (`src/convergence.py`)
`ConvergenceTracker`: judged converged if the improvement rate is below 1%
for 3 consecutive recent rounds. The logic was validated independently
before wiring it into the FL loop.

### 3. 체크포인트 (`src/checkpointing.py`)
라운드마다 저장(`save_checkpoint`), 재시작 시 자동 재개
(`resume_or_start_fresh`) — 스팟 인스턴스 회수·세션 끊김 대응.

### 3. Checkpointing (`src/checkpointing.py`)
Saves every round (`save_checkpoint`), automatically resumes on restart
(`resume_or_start_fresh`) — handles spot-instance reclamation and dropped
sessions.

### 4. FL 클라이언트 (`src/fl_client.py`)
- FedProx일 때만 `(μ/2)·‖local-global‖²` proximal term을 loss에 더함(μ=0.01)
- 첫 라운드에만 linear warmup
- `evaluate()`는 PPL(teacher-forcing 순전파)은 항상 계산하고, ROUGE-L(생성
  필요, 비쌈)은 별도의 `rouge_eval_dataset`(더 작은 샘플)에 대해서만,
  명시적으로 요청됐을 때만 계산

### 4. FL client (`src/fl_client.py`)
- Adds the `(μ/2)·‖local-global‖²` proximal term to the loss only for
  FedProx (μ=0.01)
- Linear warmup only on the first round
- `evaluate()` always computes PPL (teacher-forcing forward pass), while
  ROUGE-L (requires generation, expensive) is computed only on the separate,
  smaller `rouge_eval_dataset`, and only when explicitly requested

### 5. SCAFFOLD (`src/scaffold.py`)
Karimireddy et al. (2020) Option II의 **간소화 근사** 구현. control variate
보정을 그래디언트에 직접 더하는 방식(`scaffold_client_fit`)과, 전역
모델·전역 control variate를 함께 갱신하는 집계(`scaffold_aggregate`)로
구성 — FedAvg/FedProx의 weighted-average 집계와는 별도 경로.

### 5. SCAFFOLD (`src/scaffold.py`)
A **simplified approximation** of Karimireddy et al. (2020) Option II.
Consists of adding the control variate correction directly to the gradient
(`scaffold_client_fit`), and aggregation that jointly updates the global
model and the global control variate (`scaffold_aggregate`) — a separate
path from FedAvg/FedProx's weighted-average aggregation.

### 6. 수동 FL 라운드 루프 (`src/fl_runner.py`)
`flwr.simulation.start_simulation()`은 라운드 중간 조기종료/재개가
어려워서, 집계·평가·체크포인트·수렴판정을 직접 구현한 루프를 씁니다:

```
fit(8클라이언트) → 집계(FedAvg/FedProx 평균 or SCAFFOLD 경로)
  → PPL 평가(대표 클라이언트 1개) → 체크포인트 저장 → 수렴 판정 → (조기종료 or 다음 라운드)
```

ROUGE-L 생성은 라운드마다가 아니라 **루프가 끝난 뒤 최종 파라미터로 딱 1번만**
계산합니다 — 자기회귀 생성이 훨씬 비싸기 때문입니다.

### 6. Manual FL round loop (`src/fl_runner.py`)
`flwr.simulation.start_simulation()` makes it hard to stop early or resume
mid-loop, so we use a loop that directly implements aggregation, evaluation,
checkpointing, and convergence judgment (see the diagram above:
fit → aggregate → PPL evaluation → checkpoint → convergence check →
early-stop or next round).

ROUGE-L generation is computed **exactly once, with the final parameters,
after the loop ends** — not every round — because autoregressive generation
is far more expensive.

### 7. 검증
- `test_convergence.py`, `test_checkpointing.py`: 각 유닛 단독 검증
- `test_fl_integration.py`: FedAvg/FedProx/SCAFFOLD **세 경로 모두** 전체
  루프(fit→집계→평가→체크포인트→수렴판정)를 에러 없이 통과하는지,
  SCAFFOLD의 `local_control`이 라운드 간 유지되는지, FedProx의 proximal
  term이 실제로 동작하는지, `peak_vram_gb`/`total_latency_sec`가 제대로
  집계되는지(아래 8번)

### 7. Validation
- `test_convergence.py`, `test_checkpointing.py`: standalone validation of
  each unit
- `test_fl_integration.py`: verifies that **all three paths** — FedAvg,
  FedProx, and SCAFFOLD — pass through the full loop (fit → aggregate →
  evaluate → checkpoint → convergence judgment) without error, that
  SCAFFOLD's `local_control` is preserved across rounds, that FedProx's
  proximal term actually takes effect, and that `peak_vram_gb`/
  `total_latency_sec` are aggregated correctly (see 8 below)

### 8. 압축률×non-IID 트레이드오프 지표 보강 (지도교수 피드백 반영, `fl_runner.py`)
`fl_client.py`의 `fit()`은 원래부터 `peak_vram_gb`/`latency_sec`을
계산해서 반환하고 있었지만, `fl_runner.py`의 라운드 루프는 이 값을
`round_record`에 옮기지 않고 그냥 버리고 있었습니다 — Week 4~6에서 계획된
"압축률(LoRA/QLoRA 8bit/4bit) × non-IID(α) 트레이드오프" 분석에는 메모리·
비용 지표가 반드시 필요한데, 그동안 계산은 되고 로그에는 안 남는 상태였던
것입니다. 이번에 고친 내용:

- `round_record["peak_vram_gb"]`: 그 라운드에 참여한 클라이언트들의
  `peak_vram_gb` 평균 (FedAvg/FedProx만 — SCAFFOLD는 아래 참고)
- 최종 반환값에 `avg_peak_vram_gb`(전체 라운드 평균), `total_latency_sec`
  (`round_latency_sec` 총합) 추가 — `total_communication_bytes`와 짝을
  맞춰, 압축×α 상관분석(`compute_compression_alpha_trend`)에 그대로
  넣을 수 있게 함
- SCAFFOLD는 `scaffold_client_fit()`이 이 계측을 하지 않아 `peak_vram_gb`가
  **`None`으로 명시적으로 남습니다** — 0으로 얼버무리면 "측정했더니
  0GB"로 오인될 수 있기 때문입니다. core 압축×non-IID 분석에서는 애초에
  SCAFFOLD가 빠지므로 의도적으로 계측을 추가하지 않았습니다.

### 8. Compression × non-IID trade-off metric fix (advisor feedback, `fl_runner.py`)
`fl_client.py`'s `fit()` was already computing and returning
`peak_vram_gb`/`latency_sec`, but `fl_runner.py`'s round loop was
discarding them instead of copying them into `round_record` — the
"compression (LoRA/QLoRA 8-bit/4-bit) × non-IID (α) trade-off" analysis
planned for Weeks 4–6 needs memory/cost metrics, which were being computed
but never logged. What changed:

- `round_record["peak_vram_gb"]`: the average `peak_vram_gb` across the
  clients that participated in that round (FedAvg/FedProx only — see
  SCAFFOLD below)
- Added `avg_peak_vram_gb` (averaged across all rounds) and
  `total_latency_sec` (sum of `round_latency_sec`) to the final return
  value, mirroring `total_communication_bytes`, so they can be fed
  directly into `compute_compression_alpha_trend` for the compression × α
  correlation analysis
- SCAFFOLD's `scaffold_client_fit()` doesn't perform this measurement, so
  its `peak_vram_gb` stays **explicitly `None`** — papering over it as 0
  could be misread as "measured and it was 0GB". SCAFFOLD is already
  excluded from the core compression × non-IID analysis, so it was
  deliberately left uninstrumented.

### 설계 노트: 평가는 서버가 직접 한다 (지도교수 피드백 반영)
지도교수님 피드백("evaluation is based on the test on server, not
individual clients")을 반영해, PPL/ROUGE-L 평가는 `src/server_eval.py`의
순수 함수를 **서버 루프(`fl_runner.py`)가 직접 호출**해서 수행합니다 —
Flower의 `NumPyClient.evaluate()`(클라이언트 쪽 인터페이스)는 거치지
않습니다. `FlowerClient.evaluate()`는 Flower 인터페이스 호환성을 위해
남아있지만 내부적으로 같은 `server_eval` 함수를 재사용할 뿐, 실제
프로덕션 경로(`fl_runner.py`)는 이 메서드를 호출하지 않습니다.

평가엔 `clients[0].model`을 빌려 쓰지 않고, **어떤 클라이언트에도 속하지
않는 서버 전용 모델 인스턴스(`server_model`)**를 `run_federated_training()`
안에서 별도로 만들어 씁니다 — 8개 클라이언트가 각자 자기 모델을 GPU에
들고 있는 것과 별개로, 서버도 자기 모델을 하나 더 들고 있는 구조입니다
(로컬 시뮬레이션이라 결국 같은 프로세스/GPU를 쓰지만, 모델 인스턴스는
8+1=9개가 됩니다). 평가 시점엔 `server_model`에 그 라운드 `fit()` 결과를
**집계(aggregate)한 `global_state`**가 로드돼 있습니다. 집계 직후 8개
클라이언트는 전부 동일한 global 파라미터로 덮어써지고 동일한 공용
held-out set을 보므로, 같은 모델+같은 데이터로 8번 순전파해도 결과는
항상 같습니다(부동소수점 오차 제외) — 그래서 서버는 이 held-out을 딱
1번만 평가합니다. (반대로 `fit()`은 클라이언트마다 로컬 데이터가 달라
8번 다 필요합니다.)

### Design note: evaluation is performed by the server directly (per advisor feedback)
Reflecting advisor feedback ("evaluation is based on the test on server,
not individual clients"), PPL/ROUGE-L evaluation is performed by having
the **server loop (`fl_runner.py`) call `src/server_eval.py`'s pure
functions directly** — it does not go through Flower's
`NumPyClient.evaluate()` (a client-side interface). `FlowerClient.evaluate()`
is kept only for Flower-interface compatibility and internally reuses the
same `server_eval` functions, but the actual production path
(`fl_runner.py`) never calls that method.

Evaluation no longer borrows `clients[0].model` — `run_federated_training()`
creates a **dedicated server-only model instance (`server_model`)**,
belonging to no client. The 8 clients each already hold their own model on
the GPU; the server now holds one more of its own (still a local
simulation sharing the same process/GPU, but 8+1=9 model instances now
coexist). At evaluation time, `server_model` holds `global_state`, i.e.
that round's `fit()` results **after aggregation**. Right after
aggregation, all 8 clients are overwritten with the same global parameters
and see the same shared held-out set — running the same model on the same
data 8 times always gives the same result (aside from floating-point
error). So the server evaluates this held-out set exactly once. (Conversely, `fit()`
needs all 8, since each client has different local data.)

---

## 실행 방법 (How to Run)

```bash
pytest tests/test_convergence.py tests/test_checkpointing.py tests/test_fl_integration.py -v
```

**테스트 결과**: `pytest tests/ -m "not network"` 기준 47 passed, 2 skipped(QLoRA
4bit/8bit, GPU 필요) — Week 1~3 테스트 전체 통과. `test_data.py`(라벨 마스킹 등
`src/data.py` 유닛테스트), `test_metrics.py`(ROUGE-L/trainable params/VRAM
계측), `test_scaffold.py`(control variate 집계 수식 검증), `test_server_eval.py`
(서버 측 평가 함수 검증)가 추가돼, 그동안 어떤 테스트에서도 직접 호출되지
않던 함수들의 커버리지 공백을 메꿨습니다.

**Test results**: 47 passed, 2 skipped (QLoRA 4-bit/8-bit, requires GPU),
based on `pytest tests/ -m "not network"` — all Week 1–3 tests pass.
`test_data.py` (unit tests for `src/data.py`, incl. label masking),
`test_metrics.py` (ROUGE-L/trainable params/VRAM measurement),
`test_server_eval.py` (verifying the server-side evaluation functions), and
`test_scaffold.py` (verifying the control-variate aggregation formulas)
were added, closing a coverage gap for functions that no test had ever
called directly.

---

## 다음 주 (Week 4) 예고 (Preview of Next Week — Week 4)

**(Week 5 갱신)** 아래 예고는 원래 계획(core 6조합)이며, 실제로는 Week 5에
지도교수 피드백을 받아 압축률×Dirichlet 스윕 중심의 18조합 설계로
바뀌었습니다 — 자세한 내용은 `week5`/`week6` 브랜치 참고.

- `configs/experiment_config.yaml`: Qwen2.5-3B, 8클라이언트, 최대 10라운드 본실험 설정
- `src/evaluate.py`: 상호작용 효과 계산(`compute_interaction_effects`) — 이 프로젝트의
  핵심 산출물
- `scripts/run_experiment.py`: core 6조합 + Task 2 진단 실행 CLI
- `scripts/analyze_interaction.py`: 결과 분석 + DoRA/local-epoch 대상 자동 선정

→ `week4` 브랜치 참고.

**(Week 5 update)** The preview below reflects the original plan (core 6
combinations); it was actually revised in Week 5, per advisor feedback,
into an 18-combination design centered on a compression-rate × Dirichlet
sweep — see the `week5`/`week6` branches for details.

- `configs/experiment_config.yaml`: main-experiment config for Qwen2.5-3B, 8 clients, up to 10 rounds
- `src/evaluate.py`: interaction-effect computation (`compute_interaction_effects`) — this
  project's key deliverable
- `scripts/run_experiment.py`: CLI for running the core 6 combinations + Task 2 diagnostics
- `scripts/analyze_interaction.py`: result analysis + automatic selection of the DoRA/local-epoch target

→ See the `week4` branch.
