# Dolly-15k FL × PEFT Research Placement — Week 2 진행 보고 (Week 2 Progress Report)

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

**Research question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist as PEFT quantization (LoRA→QLoRA) becomes more aggressive?

> 이 README는 **Week 1~2 누적** 상태를 반영합니다.

> This README reflects the **cumulative Week 1–2** status.

---

## 진행 상황 요약 (Progress Summary)

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA), 단일 클라이언트 loss 감소 | ✅ 완료 |
| **Week 2** | Dolly-15k 파이프라인, 파라미터 왕복 검증, Dirichlet(α=0.5) 파티셔닝 | ✅ 완료 (이 브랜치) |
| Week 3 | FedProx/SCAFFOLD 통합 + 체크포인트/수렴 기준 | 예정 (`week3`) |
| Week 4 | Core 6조합 실행 + DoRA/local-epoch 진단 | 예정 |
| Week 5 | 상호작용 분석 + 카테고리별/수렴궤적 진단 | 예정 |
| Week 6 | 최종 리포트 | 예정 |

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, wire up PEFT (LoRA/QLoRA/DoRA), reduce loss on a single client | ✅ Done |
| **Week 2** | Dolly-15k pipeline, parameter round-trip verification, Dirichlet(α=0.5) partitioning | ✅ Done (this branch) |
| Week 3 | Integrate FedProx/SCAFFOLD + checkpoint/convergence criteria | Planned (`week3`) |
| Week 4 | Run Core 6 combinations + DoRA/local-epoch diagnostics | Planned |
| Week 5 | Interaction analysis + per-category/convergence-trajectory diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Week 1 요약 (지난 주) (Week 1 Summary (Last Week))
- `src/models.py`: LoRA(r=8)/QLoRA(NF4)/DoRA(양자화 없는 진단 대조군) 배선
- `src/communication.py`: FL 파라미터 추출 헬퍼
- `tests/test_model_wiring.py`, `scripts/run_dev_pilot.py`
- dev 모델을 `gpt2`→`tiny-random-Llama`로 교체(어텐션 구조 불일치 문제 발견 후 수정)

- `src/models.py`: wired up LoRA(r=8)/QLoRA(NF4)/DoRA (an unquantized diagnostic control)
- `src/communication.py`: FL parameter extraction helpers
- `tests/test_model_wiring.py`, `scripts/run_dev_pilot.py`
- Swapped the dev model from `gpt2`→`tiny-random-Llama` (after discovering and fixing an attention-structure mismatch)

## Week 2에서 한 일 (What Was Done in Week 2)

### 1. Dolly-15k 데이터 파이프라인 (`src/data.py`) (Dolly-15k Data Pipeline (`src/data.py`))
- `load_raw_dolly15k()`: HuggingFace `databricks/databricks-dolly-15k` 로드 (15,011개)
- `build_holdout_split(holdout_fraction=0.1)`: **카테고리별로 균등하게** 10%를 뗀 공용
  held-out 평가셋 확보 (전체 학습 시작 전 1회, 클라이언트 분배와 무관하게 고정)
- `format_prompt()` / `tokenize_example()`: `### Instruction / ### Context / ### Response`
  템플릿, 프롬프트 구간은 `labels=-100`으로 마스킹해 response만 loss에 반영
- `TokenizedDolly` Dataset, `build_client_dataloaders()`, `build_category_tagged_holdout()`

- `load_raw_dolly15k()`: loads the raw HuggingFace `databricks/databricks-dolly-15k` dataset (15,011 examples)
- `build_holdout_split(holdout_fraction=0.1)`: carves out a shared held-out eval set
  by pulling 10% **evenly per category** (once, before any training starts, fixed independent of client distribution)
- `format_prompt()` / `tokenize_example()`: `### Instruction / ### Context / ### Response`
  template; the prompt span is masked with `labels=-100` so only the response contributes to loss
- `TokenizedDolly` Dataset, `build_client_dataloaders()`, `build_category_tagged_holdout()`

### 2. Dirichlet 비IID 파티셔닝 (`src/partitioning.py`) (Dirichlet Non-IID Partitioning (`src/partitioning.py`))
- `partition_by_category(alpha=0.5, min_category_threshold=150)`: 카테고리를 기준으로
  Dirichlet(α)로 8클라이언트에 비IID 분배. 표본이 임계값(150) 미만인 카테고리는
  업샘플링 리샘플링
- `heterogeneity_score()`: 파티션이 실제로 비IID한지 정량 검증용 유틸리티
  (α가 작을수록 점수가 커야 함)

- `partition_by_category(alpha=0.5, min_category_threshold=150)`: distributes data non-IID
  across 8 clients using Dirichlet(α) keyed on category. Categories with fewer samples
  than the threshold (150) are upsampled via resampling
- `heterogeneity_score()`: a utility to quantitatively verify that a partition is actually
  non-IID (score should increase as α decreases)

### 3. 검증 (Verification)
- `tests/test_roundtrip.py`: `state_dict_to_ndarrays`↔`ndarrays_to_state_dict`가 값과
  키 순서를 모두 보존하는지 — 이게 어긋나면 서버-클라이언트 파라미터 교환 자체가 깨짐
- `tests/test_partitioning.py`: 리샘플링이 임계값을 실제로 채우는지, α=0.1 > 0.5 > 1.0
  순으로 비IID 강도가 커지는지

- `tests/test_roundtrip.py`: verifies that `state_dict_to_ndarrays`↔`ndarrays_to_state_dict`
  preserve both values and key order — if this breaks, the server-client parameter exchange itself is broken
- `tests/test_partitioning.py`: verifies that resampling actually fills the threshold, and that
  non-IID strength increases in the order α=0.1 > 0.5 > 1.0

### 설계 노트 (Design Notes)
- Held-out은 **전역 공유**(클라이언트별로 나누지 않음) — 이유는 Week 6 한계 항목 참고.
- α=0.5는 **고정값**입니다. 원래 계획서 초안에는 α 스윕(1.0/0.5/0.1)이 있었지만,
  6주 스코프 안에서 핵심 질문(교정×양자화 상호작용)에 직접 필요하지 않아 제외했습니다.
  **(Week 5 갱신)** 지도교수 피드백으로 이 α 스윕은 다시 core 설계에
  포함됐습니다 — 자세한 내용은 `week5`/`week6` 브랜치 참고.

- The held-out set is **shared globally** (not split per client) — see the Week 6 limitations
  section for the rationale.
- α=0.5 is a **fixed value**. The original draft plan included an α sweep (1.0/0.5/0.1), but it
  was dropped within the 6-week scope since it is not directly required for the core question
  (correction × quantization interaction). **(Week 5 update)** This α sweep was reinstated into
  the core design following advisor feedback — see the `week5`/`week6` branches for details.

---

## 실행 방법 (How to Run)

```bash
pytest tests/test_model_wiring.py tests/test_roundtrip.py tests/test_partitioning.py -v
```

**테스트 결과**: 9 passed, 2 skipped(QLoRA 4bit/8bit, GPU 필요)

**Test results**: 9 passed, 2 skipped (QLoRA 4-bit/8-bit, requires GPU)

---

## 다음 주 (Week 3) 예고 (Preview: Next Week (Week 3))

- `src/fl_client.py`: FedProx proximal term, warmup, VRAM/latency/PPL/ROUGE-L
- `src/scaffold.py`: SCAFFOLD control variate (Option II 간소화)
- `src/fl_runner.py`: 수동 FL 라운드 루프(집계/체크포인트/수렴 판정)
- `src/convergence.py`, `src/checkpointing.py`, `src/metrics.py`

- `src/fl_client.py`: FedProx proximal term, warmup, VRAM/latency/PPL/ROUGE-L
- `src/scaffold.py`: SCAFFOLD control variate (a simplified Option II)
- `src/fl_runner.py`: a manual FL round loop (aggregation/checkpointing/convergence determination)
- `src/convergence.py`, `src/checkpointing.py`, `src/metrics.py`

→ `week3` 브랜치 참고.

→ See the `week3` branch.
