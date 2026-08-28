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
| Week 2 | Dolly-15k 파이프라인, Dirichlet(α=1) 파티셔닝 | ✅ 완료 |
| Week 3 | FedProx/SCAFFOLD 통합, 체크포인트/수렴 기준 | ✅ 완료 |
| Week 4 | Core 18조합(3압축×2FL×3α) 실행 스크립트 + 압축률×non-IID 상관관계 분석 함수 | ✅ 완료 |
| **Week 5** | 지도교수 피드백 반영 설계 개정 + 상관관계/선정 로직 테스트 완료 | ✅ 완료 (이 브랜치) |
| Week 6 | 최종 리포트 | 예정 (`week6`) |

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, PEFT wiring | ✅ Done |
| Week 2 | Dolly-15k pipeline, Dirichlet(α=1) partitioning | ✅ Done |
| Week 3 | FedProx/SCAFFOLD integration, checkpoint/convergence criteria | ✅ Done |
| Week 4 | Core 18-combination (3 compression × 2 FL × 3 alpha) run script + compression×non-IID correlation analysis functions | ✅ Done |
| **Week 5** | Redesign per advisor feedback + correlation/selection logic tests complete | ✅ Done (this branch) |
| Week 6 | Final report | Planned (`week6`) |

---

## Week 1~4 요약 (Week 1–4 Summary)
- **Week 1**: PEFT 배선(LoRA/QLoRA/DoRA)
- **Week 2**: Dolly-15k 파이프라인, Dirichlet(α=1) 파티셔닝
- **Week 3**: FedProx/SCAFFOLD, 체크포인트, 수렴 기준, FL 통합 테스트
- **Week 4**: `configs/experiment_config.yaml`, `src/evaluate.py`(압축률×α 상관관계 계산),
  `scripts/run_experiment.py`, `scripts/analyze_interaction.py`

- **Week 1**: PEFT wiring (LoRA/QLoRA/DoRA)
- **Week 2**: Dolly-15k pipeline, Dirichlet(α=1) partitioning
- **Week 3**: FedProx/SCAFFOLD, checkpoints, convergence criteria, FL integration tests
- **Week 4**: `configs/experiment_config.yaml`, `src/evaluate.py` (compression×alpha correlation),
  `scripts/run_experiment.py`, `scripts/analyze_interaction.py`

## Week 5에서 한 일 (What Was Done in Week 5)

### 지도교수 피드백에 따른 설계 개정 (Redesign per Advisor Feedback)
지도교수로부터 "이 연구의 중점을 압축률과 non-IID 강도의 상관관계로 두는
게 좋겠다"는 피드백을 받아, core 설계를 3FL×2PEFT(6조합)에서 **3압축
(LoRA/QLoRA 8bit/QLoRA 4bit) × 2FL(FedAvg/FedProx) × 3 Dirichlet
α(0.1/1/10) = 18조합**으로 개정했습니다. SCAFFOLD는 core에서
제외했지만 `src/scaffold.py`와 관련 테스트는 그대로 유지했습니다. 모델도
Qwen2.5-3B에서 **Qwen2.5-1.5B-Instruct**로 축소했습니다(지도교수 승인,
GPU 비용 절감). 자세한 배경은 `week4` 브랜치의 "설계 노트" 절 참고.

Following advisor feedback that "this project's focus should be the
correlation between compression rate and non-IID intensity," the core
design was revised from 3 FL × 2 PEFT (6 combinations) to **3
compression levels (LoRA/QLoRA-8bit/QLoRA-4bit) × 2 FL algorithms
(FedAvg/FedProx) × 3 Dirichlet alphas (0.1/1/10) = 18 combinations**.
SCAFFOLD is dropped from the core design, though `src/scaffold.py` and
its tests are kept as-is. The model was also downsized from Qwen2.5-3B to
**Qwen2.5-1.5B-Instruct** (advisor-approved, to cut GPU cost). See the
"Design Note" section on the `week4` branch for the full background.

### `tests/test_evaluate.py` 확장 (Extended)
- `compute_compression_alpha_trend`(새 핵심 산출물) 테스트 3개: 페널티 값이
  올바르게 계산되는지, non-IID가 강해질수록 페널티가 커지면 상관계수가
  음수인지, 페널티가 α와 무관하면(분산 0) 상관계수가 NaN이 아니라 0.0으로
  나오는지(numpy가 0으로 나누는 문제를 명시적으로 처리)
- 기존 `compute_interaction_effects`/`select_largest_interaction_fl_algorithm`
  테스트는 그대로 유지 — 함수 자체는 안 지웠으므로 여전히 유효
- `tests/test_model_wiring.py`에 QLoRA 8bit 배선 테스트 추가(압축률 스윕용)

이 시점부로 **전체 테스트 스위트가 모두 갖춰졌습니다** — `pytest tests/`가
Week 1~5에서 만든 모든 모듈(개정된 설계 포함)을 커버합니다.

- 3 tests for `compute_compression_alpha_trend` (the new key deliverable):
  that the penalty values are computed correctly, that the correlation is
  negative when the penalty grows as non-IID intensifies, and that the
  correlation returns 0.0 (not NaN) when the penalty is independent of
  alpha (zero variance) — explicitly handling numpy's divide-by-zero
- The existing `compute_interaction_effects`/
  `select_largest_interaction_fl_algorithm` tests are kept as-is — the
  functions themselves were not removed, so they remain valid
- Added a QLoRA 8-bit wiring test to `tests/test_model_wiring.py` (for the
  compression-rate sweep)

As of this point, **the full test suite is complete** — `pytest tests/`
covers every module built across Week 1–5, including the revised design.

### Subtask 2.3 분석은 코드가 아니라 실행 시점의 작업 (Subtask 2.3 Analysis Is a Runtime Task, Not Code)
카테고리별 성능 분해(`per_category_breakdown`)는 추가 코드 없이,
`results/logs/*_rounds.jsonl`에 이미 기록된 데이터에서 바로 계산됩니다 —
실제 GPU 실행(Week 4 스크립트)이 끝난 뒤 노트북/스크립트에서 이 함수들을
직접 호출하는 방식이라, 실행 결과가 나오기 전까지는 추가로 커밋할 코드가
없습니다.

Per-category performance breakdown (`per_category_breakdown`) is computed
directly from data already logged in `results/logs/*_rounds.jsonl`,
without any additional code — these functions are called directly from a
notebook/script after the actual GPU run (Week 4 scripts) finishes, so
there is no additional code to commit until run results are available.

---

## 실행 방법 (How to Run)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**테스트 결과**: **28 passed, 2 skipped**(QLoRA 4bit/8bit, GPU 필요) — 전체 스위트.

**Test results**: **28 passed, 2 skipped** (QLoRA 4-bit/8-bit, requires GPU) — full suite.

---

## 다음 주 (Week 6) 예고 (Next Week (Week 6) Preview)

- 최종 리포트 작성: 연구질문/설계 개정 배경/결과 해석/한계(SCAFFOLD를 core에서
  제외, single-seed, 공용 held-out) 정리
- 발표 준비

- Write final report: research question/redesign background/results interpretation/
  limitations (SCAFFOLD dropped from core, single-seed, shared held-out)
- Prepare presentation

→ `week6` 브랜치 참고 (코드는 이 브랜치와 동일, README만 최종 리포트로 교체).

→ See the `week6` branch (code identical to this branch; only the README is
replaced with the final report).
