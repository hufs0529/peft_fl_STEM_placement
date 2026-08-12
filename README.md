# Dolly-15k FL × PEFT Research Placement — Week 2 진행 보고

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

> 이 README는 **Week 1~2 누적** 상태를 반영합니다.

---

## 진행 상황 요약

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선(LoRA/QLoRA/DoRA), 단일 클라이언트 loss 감소 | ✅ 완료 |
| **Week 2** | Dolly-15k 파이프라인, 파라미터 왕복 검증, Dirichlet(α=0.5) 파티셔닝 | ✅ 완료 (이 브랜치) |
| Week 3 | FedProx/SCAFFOLD 통합 + 체크포인트/수렴 기준 | 예정 (`week3`) |
| Week 4 | Core 6조합 실행 + DoRA/local-epoch 진단 | 예정 |
| Week 5 | 상호작용 분석 + 카테고리별/수렴궤적 진단 | 예정 |
| Week 6 | 최종 리포트 | 예정 |

---

## Week 1 요약 (지난 주)
- `src/models.py`: LoRA(r=8)/QLoRA(NF4)/DoRA(양자화 없는 진단 대조군) 배선
- `src/communication.py`: FL 파라미터 추출 헬퍼
- `tests/test_model_wiring.py`, `scripts/run_dev_pilot.py`
- dev 모델을 `gpt2`→`tiny-random-Llama`로 교체(어텐션 구조 불일치 문제 발견 후 수정)

## Week 2에서 한 일

### 1. Dolly-15k 데이터 파이프라인 (`src/data.py`)
- `load_raw_dolly15k()`: HuggingFace `databricks/databricks-dolly-15k` 로드 (15,011개)
- `build_holdout_split(holdout_fraction=0.1)`: **카테고리별로 균등하게** 10%를 뗀 공용
  held-out 평가셋 확보 (전체 학습 시작 전 1회, 클라이언트 분배와 무관하게 고정)
- `format_prompt()` / `tokenize_example()`: `### Instruction / ### Context / ### Response`
  템플릿, 프롬프트 구간은 `labels=-100`으로 마스킹해 response만 loss에 반영
- `TokenizedDolly` Dataset, `build_client_dataloaders()`, `build_category_tagged_holdout()`

### 2. Dirichlet 비IID 파티셔닝 (`src/partitioning.py`)
- `partition_by_category(alpha=0.5, min_category_threshold=150)`: 카테고리를 기준으로
  Dirichlet(α)로 8클라이언트에 비IID 분배. 표본이 임계값(150) 미만인 카테고리는
  업샘플링 리샘플링
- `heterogeneity_score()`: 파티션이 실제로 비IID한지 정량 검증용 유틸리티
  (α가 작을수록 점수가 커야 함)

### 3. 검증
- `tests/test_roundtrip.py`: `state_dict_to_ndarrays`↔`ndarrays_to_state_dict`가 값과
  키 순서를 모두 보존하는지 — 이게 어긋나면 서버-클라이언트 파라미터 교환 자체가 깨짐
- `tests/test_partitioning.py`: 리샘플링이 임계값을 실제로 채우는지, α=0.1 > 0.5 > 1.0
  순으로 비IID 강도가 커지는지

### 설계 노트
- Held-out은 **전역 공유**(클라이언트별로 나누지 않음) — 이유는 Week 6 한계 항목 참고.
- α=0.5는 **고정값**입니다. 원래 계획서 초안에는 α 스윕(1.0/0.5/0.1)이 있었지만,
  6주 스코프 안에서 핵심 질문(교정×양자화 상호작용)에 직접 필요하지 않아 제외했습니다.

---

## 실행 방법

```bash
pytest tests/test_model_wiring.py tests/test_roundtrip.py tests/test_partitioning.py -v
```

**테스트 결과**: 9 passed, 1 skipped(QLoRA, GPU 필요)

---

## 다음 주 (Week 3) 예고

- `src/fl_client.py`: FedProx proximal term, warmup, VRAM/latency/PPL/ROUGE-L
- `src/scaffold.py`: SCAFFOLD control variate (Option II 간소화)
- `src/fl_runner.py`: 수동 FL 라운드 루프(집계/체크포인트/수렴 판정)
- `src/convergence.py`, `src/checkpointing.py`, `src/metrics.py`

→ `week3` 브랜치 참고.
