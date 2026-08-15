# Dolly-15k FL × PEFT Research Placement — Week 5 진행 보고 (Progress Report)

**연구질문**: FL 드리프트 교정(FedAvg→FedProx→SCAFFOLD)의 이점이 PEFT 양자화(LoRA→QLoRA)가
심해져도 유지되는가?

**Research Question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist as PEFT quantization (LoRA→QLoRA) becomes more aggressive?

> 이 README는 **Week 1~5 누적** 상태를 반영합니다. 코드는 이 시점 이후
> `week6`에서 최종 리포트 문서화 외에는 추가 변경이 없습니다 — 즉
> **`week5`가 사실상 코드 완성 시점**입니다.
>
> This README reflects the **cumulative Week 1–5** state. No further code
> changes occur after this point except final report documentation in
> `week6` — in other words, **`week5` is effectively the code-complete
> point**.

---

## 진행 상황 요약 (Progress Summary)

| 주차 | 목표 | 상태 |
|---|---|---|
| Week 1 | 모델+GPU 환경 확인, PEFT 배선 | ✅ 완료 |
| Week 2 | Dolly-15k 파이프라인, Dirichlet(α=0.5) 파티셔닝 | ✅ 완료 |
| Week 3 | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준 | ✅ 완료 |
| Week 4 | Core 6조합 실행 스크립트 + 상호작용 분석 함수 | ✅ 완료 |
| **Week 5** | 상호작용/선정 로직 테스트, 분석 파이프라인 검증 완료 | ✅ 완료 (이 브랜치) |
| Week 6 | 최종 리포트 | 예정 (`week6`) |

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, PEFT wiring | ✅ Done |
| Week 2 | Dolly-15k pipeline, Dirichlet(α=0.5) partitioning | ✅ Done |
| Week 3 | FedProx/SCAFFOLD integration, checkpoint/convergence criteria | ✅ Done |
| Week 4 | Core 6-combination run script + interaction analysis functions | ✅ Done |
| **Week 5** | Interaction/selection logic tests, analysis pipeline verification complete | ✅ Done (this branch) |
| Week 6 | Final report | Planned (`week6`) |

---

## Week 1~4 요약 (Week 1–4 Summary)
- **Week 1**: PEFT 배선(LoRA/QLoRA/DoRA)
- **Week 2**: Dolly-15k 파이프라인, Dirichlet(α=0.5) 파티셔닝
- **Week 3**: FedProx/SCAFFOLD, 체크포인트, 수렴 기준, FL 통합 테스트
- **Week 4**: `configs/experiment_config.yaml`, `src/evaluate.py`(상호작용 계산),
  `scripts/run_experiment.py`, `scripts/analyze_interaction.py`

- **Week 1**: PEFT wiring (LoRA/QLoRA/DoRA)
- **Week 2**: Dolly-15k pipeline, Dirichlet(α=0.5) partitioning
- **Week 3**: FedProx/SCAFFOLD, checkpoints, convergence criteria, FL integration tests
- **Week 4**: `configs/experiment_config.yaml`, `src/evaluate.py` (interaction computation),
  `scripts/run_experiment.py`, `scripts/analyze_interaction.py`

## Week 5에서 한 일 (What Was Done in Week 5)

### `tests/test_evaluate.py` 추가 (Added)
Week 4에서 만든 분석 함수들이 실제로 의도대로 동작하는지 단위 테스트로 고정:

- `test_compute_interaction_effects_sign_and_shape`: 가상의 6조합 성능값으로
  delta·interaction 계산이 부호/값 모두 맞는지
- `test_select_largest_interaction_fl_algorithm_picks_bigger_magnitude`:
  |interaction|이 더 큰 쪽(FedProx vs SCAFFOLD)을 정확히 고르는지
- `test_select_best_performing_combination_prefers_lowest_perplexity`:
  6조합 중 성능(val_perplexity) 최저 조합을 정확히 고르는지
- `test_per_category_breakdown`, `test_client_fairness_variance_*`,
  `test_total_communication_cost`: Subtask 2.3 유틸리티 검증

이 시점부로 **전체 테스트 스위트가 모두 갖춰졌습니다** — `pytest tests/`가
Week 1~5에서 만든 모든 모듈을 커버합니다.

Unit tests that lock in the intended behavior of the analysis functions built in Week 4:

- `test_compute_interaction_effects_sign_and_shape`: checks that the delta/interaction
  computation is correct in both sign and value, using synthetic performance values
  for the 6 combinations
- `test_select_largest_interaction_fl_algorithm_picks_bigger_magnitude`: checks that the
  side with the larger |interaction| (FedProx vs SCAFFOLD) is picked correctly
- `test_select_best_performing_combination_prefers_lowest_perplexity`: checks that the
  combination with the lowest performance (val_perplexity) among the 6 is picked correctly
- `test_per_category_breakdown`, `test_client_fairness_variance_*`,
  `test_total_communication_cost`: verify Subtask 2.3 utilities

As of this point, **the full test suite is complete** — `pytest tests/`
covers every module built across Week 1–5.

### Subtask 2.3 분석은 코드가 아니라 실행 시점의 작업 (Subtask 2.3 Analysis Is a Runtime Task, Not Code)
카테고리별 성능 분해(`per_category_breakdown`)와 수렴 궤적 비교는 추가
코드 없이, `results/logs/*_rounds.jsonl`에 이미 기록된 데이터에서 바로
계산됩니다 — 실제 GPU 실행(Week 4 스크립트)이 끝난 뒤 노트북/스크립트에서
이 함수들을 직접 호출하는 방식이라, 실행 결과가 나오기 전까지는 추가로
커밋할 코드가 없습니다.

Per-category performance breakdown (`per_category_breakdown`) and convergence
trajectory comparison are computed directly from data already logged in
`results/logs/*_rounds.jsonl`, without any additional code — these functions
are called directly from a notebook/script after the actual GPU run (Week 4
scripts) finishes, so there is no additional code to commit until run results
are available.

---

## 실행 방법 (How to Run)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**테스트 결과**: **25 passed, 1 skipped**(QLoRA, GPU 필요) — 전체 스위트.

**Test results**: **25 passed, 1 skipped** (QLoRA, requires GPU) — full suite.

---

## 다음 주 (Week 6) 예고 (Next Week (Week 6) Preview)

- 최종 리포트 작성: 연구질문/설계/결과 해석/한계(SCAFFOLD 간소화, single-seed,
  공용 held-out, 단일 α) 정리
- 발표 준비

- Write final report: research question/design/results interpretation/limitations
  (SCAFFOLD simplification, single-seed, shared held-out, single α)
- Prepare presentation

→ `week6` 브랜치 참고 (코드는 이 브랜치와 동일, README만 최종 리포트로 교체).

→ See the `week6` branch (code identical to this branch; only the README is
replaced with the final report).
