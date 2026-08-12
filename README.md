# Dolly-15k FL × PEFT Research Placement

6주 연구 실습 프로젝트 코드베이스. 연구계획서 *"Does Drift-Correction
Sophistication Interact with PEFT Quantisation Noise in Federated LLM
Fine-Tuning?"*를 그대로 코드로 구현했습니다.

---

## 0. 이 프로젝트가 묻는 질문 (연구계획서 Aims, 코드와의 대응)

| Aim | 질문 요약 | 답을 만드는 코드 |
|---|---|---|
| **Aim 1** (Task 1) | 교정 정교함(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 압축이 LoRA에서 QLoRA로 심해져도 일관되게 유지되는가? 3×2 factorial로 검증. | `src/fl_client.py`(FedProx proximal term), `src/scaffold.py`(SCAFFOLD), `src/evaluate.py::compute_interaction_effects`(상호작용 계산) |
| **Aim 2** (Task 2) | 발견된 상호작용이 "양자화 노이즈가 교정 메커니즘을 방해하기 때문"인지, "압축 자체" 때문인지 진단한다. | `src/models.py`(DoRA 대조군), `src/evaluate.py::select_largest_interaction_fl_algorithm`, `src/evaluate.py::per_category_breakdown` |

**핵심 설계 원칙**: 이 코드베이스는 6개 조합(3 FL × 2 PEFT)의 core 실험으로
"교정 정교함 × 압축 강도"의 2×3 격자를 직접 채우고, Task 2는 이 격자가
보여준 상호작용을 **왜** 나타나는지 설명하는 데 필요한 최소한의 진단
실험(DoRA 대조군, local-epoch 강건성 점검) 두 가지로만 한정합니다 —
관련 없는 축(Prompt Tuning, α 스윕, 부분 참여, LLM-judge, 중앙집중형
베이스라인)은 이번 연구계획서 범위 밖이므로 포함하지 않습니다.

---

## 1. 폴더 구조

```
dolly15k-fl-peft-v2/
├── README.md                     # 이 파일
├── requirements.txt
├── configs/
│   ├── dev_config.yaml           # Week 1~3 검증용 (gpt2, 2클라이언트, 2라운드)
│   └── experiment_config.yaml    # Week 4 본실험용 (Qwen2.5-3B, 8클라이언트, 10라운드)
├── src/                          # dev/experiment 공용 핵심 로직
│   ├── models.py                   # LoRA/QLoRA(core), DoRA(Task 2 진단 대조군) 래핑
│   ├── data.py                     # Dolly-15k 로딩, 토크나이징, held-out 스플릿
│   ├── partitioning.py             # Dirichlet(alpha=0.5) 비IID + 카테고리 리샘플링
│   ├── communication.py            # FL 파라미터 왕복 + payload 크기 계산
│   ├── fl_client.py                # 공통 클라이언트 (FedProx proximal term, warmup, VRAM/latency, ROUGE-L)
│   ├── fl_runner.py                # 수동 라운드 루프 — 수렴기준/체크포인트/로깅/W&B 전부 배선
│   ├── scaffold.py                 # SCAFFOLD 실제 구현 (control variate)
│   ├── convergence.py              # 3라운드 연속 <1% 개선 시 조기종료
│   ├── checkpointing.py            # 라운드별 저장/재개
│   ├── metrics.py                  # VRAM, latency, ROUGE-L, trainable param count
│   └── evaluate.py                 # 카테고리별 분해, client fairness, 상호작용 효과 계산
├── tests/                        # 아래 "테스트 구성" 절 참고
├── scripts/
│   ├── run_dev_pilot.py            # Week 1: 실제 모델 단일 클라이언트 검증
│   ├── run_experiment.py           # Task 1 core 6조합 + Task 2 진단 실행 진입점
│   └── analyze_interaction.py      # Subtask 1.3/2.1/2.2: 상호작용 분석 + 진단 대상 선정
└── results/
    ├── checkpoints/{run_name}/round_NNN.pt   # 라운드별 체크포인트 (git 추적 제외)
    ├── logs/{run_name}_rounds.jsonl          # 라운드별 전체 지표
    ├── logs/{run_name}_generations.jsonl     # 마지막 생성 라운드의 예측/참조 (Subtask 2.3 입력)
    └── figures/                              # Week 5 시각화 출력
```

---

## 2. Task 1 core 설계가 코드에 어떻게 반영되는가

연구계획서 Subtask 1.1/1.2의 core 설계(**3 FL × 2 PEFT = 6조합**)는 하나의
스크립트(`run_experiment.py`)에 `--peft`, `--fl` 두 인자로 압축되어 있습니다.
즉 아래 6개 명령이 core 전체입니다:

```bash
python scripts/run_experiment.py --peft lora  --fl fedavg
python scripts/run_experiment.py --peft lora  --fl fedprox
python scripts/run_experiment.py --peft lora  --fl scaffold
python scripts/run_experiment.py --peft qlora --fl fedavg
python scripts/run_experiment.py --peft qlora --fl fedprox
python scripts/run_experiment.py --peft qlora --fl scaffold
```

6개가 모두 끝나면:

```bash
python scripts/analyze_interaction.py
```

를 실행합니다. 이 스크립트는 세 가지를 계산합니다:

1. **Subtask 1.3 상호작용 효과** — 각 PEFT(LoRA/QLoRA)에서 FedProx/SCAFFOLD가
   FedAvg 대비 얼마나 개선하는지(delta)를 구하고, 그 delta가 QLoRA에서
   LoRA 대비 얼마나 달라지는지(`interaction = delta_qlora - delta_lora`)를
   계산합니다. 이것이 연구질문("교정 정교함의 이점이 양자화와 결합해도
   유지되는가")에 대한 직접적인 정량적 답입니다(`src/evaluate.py::compute_interaction_effects`).
2. **Subtask 2.1 대상 선정** — `|interaction|`이 더 큰 FL 알고리즘(FedProx
   또는 SCAFFOLD)을 골라, 이후 DoRA 진단 실험과 짝짓습니다.
3. **Subtask 2.2 대상 선정** — 6조합 중 task performance가 가장 좋았던
   조합을 골라, local_epochs=5 강건성 점검 대상으로 삼습니다.

### Task 2 진단 실행 순서

| 순서 | 무엇을 검증하는가 | 명령 |
|---|---|---|
| 1 (Subtask 2.1) | DoRA × `<largest_interaction_fl>` — 양자화 노이즈 vs 압축 자체의 인과관계 분리 | `run_experiment.py --peft dora --fl <largest_interaction_fl>` |
| 2 (Subtask 2.2) | Local epoch 1 vs 5 강건성 점검 (best-performing core 조합에 적용) | `run_experiment.py --peft <best_peft> --fl <best_fl> --local-epochs 5` |

`<largest_interaction_fl>`/`<best_peft>`/`<best_fl>`은
`analyze_interaction.py` 출력에서 그대로 복사해 쓰면 됩니다.

Subtask 2.3(카테고리별 분해, 수렴 궤적 비교)은 추가 GPU 비용 없이
`src/evaluate.py`의 `per_category_breakdown`과
`results/logs/*_rounds.jsonl`에 이미 기록된 라운드별 지표에서 바로
계산합니다.

---

## 3. 각 모듈이 연구계획서의 어느 부분과 대응하는가

### `src/models.py` — Subtask 1.1, 2.1
- **core**: LoRA(r=8, α=16, dropout=0.05), QLoRA(NF4 4bit)
- **diagnostic**: DoRA(`use_dora=True`, 양자화 없음) — LoRA와 압축률은
  같지만 양자화가 없는 대조군. Task 1에서 상호작용이 가장 컸던 FL
  알고리즘과만 짝지어 1회 실행합니다.
- 다른 PEFT 방식(Prompt Tuning, Adapter Tuning 등)은 이번 연구계획서의
  범위 밖이므로 포함하지 않았습니다 — `lora`/`qlora`/`dora` 이외의 값을
  넘기면 `ValueError`가 나는 것도 의도된 동작이며,
  `tests/test_model_wiring.py`에서 이를 확인합니다.

### `src/scaffold.py` — Subtask 1.1
FedAvg/FedProx는 Flower의 표준 weighted-average 집계로 충분하지만, SCAFFOLD는
클라이언트별 control variate를 별도로 유지·통신해야 해서 `fl_runner.py`가 이
파일의 `scaffold_client_fit()`/`scaffold_aggregate()`를 직접 호출하는 별도
경로로 처리합니다. **간소화 구현(Option II 근사)**이라는 점을 최종 리포트의
Limitations에 반드시 명시하세요.

### `src/fl_client.py` — Subtask 1.1, 1.2
- FedProx일 때만 `(mu/2)*||local-global||^2` proximal term이 loss에 추가됩니다
  (`is_fedprox` 분기, μ=0.01).
- 첫 라운드(`server_round==1`)에만 linear warmup이 적용됩니다.
- `track_vram_and_latency()`로 감싸서 Peak VRAM/Training Latency를 자동 측정합니다.
- `evaluate()`는 매 라운드 PPL(teacher-forcing 순전파)만 계산하고, ROUGE-L
  (자기회귀 생성이 필요해 훨씬 비쌈)은 `run_generation_metrics=True`일
  때만 별도의 `rouge_eval_dataset`(카테고리 층화 샘플)에 대해 계산합니다
  — 언제 이 플래그가 켜지는지는 `fl_runner.py`가 결정합니다(아래).

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

### `src/evaluate.py` — Subtask 1.3, 2.1, 2.2, 2.3
- `compute_interaction_effects`: **연구질문의 직접적인 답**을 계산하는 함수.
  core 6조합의 성능값에서 상호작용 효과(`interaction_fedprox`,
  `interaction_scaffold`)를 산출합니다. `performance_field`로
  `val_perplexity`(성능)와 `rounds_run`(수렴 속도, 완전 무료로 이미
  로깅된 값) 둘 다에 대해 계산해 두 지표가 같은 방향을 가리키는지
  교차검증합니다(`scripts/analyze_interaction.py`).
- `select_largest_interaction_fl_algorithm`: Subtask 2.1에서 DoRA와 짝지을
  FL 알고리즘을 고릅니다.
- `select_best_performing_combination`: Subtask 2.2에서 local_epochs=5
  점검 대상을 고릅니다.
- `per_category_breakdown`, `total_communication_cost`: Subtask 2.3
  분석에서 그대로 불러 쓰는 함수들. 추가 GPU 비용 없이 이미 로깅된
  데이터에서 계산됩니다.
- `client_fairness_variance`: 순수 유틸리티 함수로만 남아있습니다 —
  현재 held-out set이 클라이언트 간 전역 공유라 파이프라인에서는 호출하지
  않습니다(항상 동일값이라 무의미). 아래 "알려진 한계" 참고.

---

## 4. 워크플로우 (6주 타임라인과 매핑)

| 주차 | 실행 명령 | 검증 대상 |
|---|---|---|
| Week 1 | `pytest tests/test_model_wiring.py` → `python scripts/run_dev_pilot.py --stage single_client` | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA), 단일 클라이언트 loss 감소 |
| Week 2 | `pytest tests/test_roundtrip.py tests/test_partitioning.py` | Dolly-15k 파이프라인, 파라미터 왕복, Dirichlet(α=0.5) 파티셔닝 검증 |
| Week 3 | `pytest tests/test_fl_integration.py tests/test_convergence.py tests/test_checkpointing.py` | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준, 축소 규모 6조합 예행연습 |
| Week 4 | `run_experiment.py` × 6(core) → `analyze_interaction.py` → DoRA 진단 + local-epoch 강건성 점검 | Subtask 1.2 본 실험 + Subtask 2.1/2.2 진단 |
| Week 5 | `pytest tests/test_evaluate.py` → 분석 노트북에서 `src/evaluate.py` 함수 직접 호출 | Subtask 1.3 상호작용 분석, Subtask 2.3 카테고리별/수렴궤적 진단 |
| Week 6 | 리포트 작성 (SCAFFOLD 간소화·single-seed 한계 명시) + 발표 | — |

**전체 테스트를 한 번에 돌리려면**:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## 5. 테스트 구성

| 파일 | 무엇을 검증하는가 | 대응 단계 |
|---|---|---|
| `test_model_wiring.py` | LoRA/QLoRA/DoRA 배선, 정의되지 않은 PEFT type이 `ValueError`를 내는지 | Week 1 |
| `test_roundtrip.py` | FL 파라미터 송수신 시 값/순서 보존 | Week 2 |
| `test_partitioning.py` | 최소 카테고리 임계값 리샘플링, α가 작을수록 실제로 더 비IID한지 | Week 2 |
| `test_fl_integration.py` | FedAvg/FedProx/SCAFFOLD 세 경로 모두 전체 루프 통과, SCAFFOLD의 `local_control` 라운드 간 유지, FedProx fit() 정상 동작 | Week 3 |
| `test_convergence.py` | 3라운드 연속 <1% 개선 시 실제로 멈추는지, 큰 폭 개선 중에는 안 멈추는지 | Week 3 |
| `test_checkpointing.py` | 저장 후 재개 시 라운드/상태가 정확히 복원되는지 | Week 3 |
| `test_evaluate.py` | 카테고리별 분해, fairness variance, **상호작용 효과 계산**, **가장 큰 상호작용 FL 알고리즘 선정**, **최고 성능 조합 선정** | Week 4~5 |

---

## 6. 실행 환경

- **Week 1~3 (개발/디버깅)**: Google Colab Pro — compute unit 소모가 적은
  미니어처 검증 위주라 기본 구독만으로 충분합니다.
- **Week 4 (본 실험)**: 예약형/스팟 단일 GPU 인스턴스(AWS g5.xlarge 또는
  Lambda Labs A100) — `src/checkpointing.py`가 매 라운드 저장하므로 스팟
  인스턴스가 회수되어도 안전하게 재개할 수 있습니다.

```bash
pip install -r requirements.txt
# bitsandbytes는 GPU 환경에서만 정상 설치/동작합니다.
```

---

## 7. 알려진 한계 (Week 6 리포트 Limitations에 반영할 것)

- **SCAFFOLD는 Option II의 간소화 근사 구현**입니다.
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
- 비IID 강도는 **α=0.5 단일 지점만** 테스트합니다 — 다른 α 값에서 상호작용
  패턴이 달라질 수 있다는 점을 리포트에서 명시하는 것을 권장합니다.
- 코드 전체는 **문법 검증만 마쳤고 실제 GPU 환경에서 아직 실행되지 않았습니다.**
  Week 1 첫날 `pytest tests/`부터 돌려서 실제 동작을 확인하세요.
