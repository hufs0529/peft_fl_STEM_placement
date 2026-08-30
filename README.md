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

### `compute_compression_alpha_trend` 보강: ROUGE-L 부호, qlora_8bit 비교, 카테고리별 취약도 (Hardening `compute_compression_alpha_trend`: ROUGE-L Sign, qlora_8bit Comparison, Per-Category Vulnerability)

`compute_compression_alpha_trend`는 원래 "낮을수록 좋음"(PPL) 가정으로
페널티 부호를 정했는데, 그대로 ROUGE-L(높을수록 좋음)을 넣으면 "압축이
성능을 해쳤다"는 의미가 반대로 뒤집히는 문제가 있었습니다. 세 가지를
보강했습니다:

- `higher_is_better=False`(기본, 하위호환) 인자 추가 — ROUGE-L처럼
  높을수록 좋은 지표는 `higher_is_better=True`로 넘기면 부호 변환 없이
  그대로 쓸 수 있습니다.
- `compression="qlora_4bit"`(기본) 인자 추가 — `"qlora_8bit"`을 넘겨서
  같은 함수를 한 번 더 호출하면, "4bit가 8bit보다 non-IID에 더
  민감한가"(dose-response)를 두 상관계수로 비교할 수 있습니다.
- `per_category_compression_penalty()` 신규 추가 — 같은 상관관계 분석을
  카테고리 단위로 쪼개서, 어떤 태스크 카테고리가 압축×non-IID 상호작용에
  특히 취약한지 상대 페널티의 z-score로 스크리닝합니다(카테고리가 8종뿐이라
  엄밀한 유의성 검정이 아니라 상대적 순위 매기기 용도).

`scripts/analyze_interaction.py`는 이제 4bit/8bit 트렌드를 나란히,
PPL뿐 아니라 ROUGE-L 기준으로도 출력합니다. `scripts/run_experiment.py`에는
`--num-rounds` 오버라이드를 추가했습니다 — 본 18조합을 다 돌리기 전에
가장 어려운 조합(`qlora_4bit`, `α=0.1`)을 넉넉한 캡(예: 30)으로 먼저
파일럿 실행해 실제 `converged_round`를 확인하고, 그 값 + 여유분을
`experiment_config.yaml`의 `federated.num_rounds`에 반영하는 식으로
상한을 역산하기 위함입니다:

```bash
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 0.1 --num-rounds 30
```

이 파일럿 run은 core 18조합 중 하나(`qlora_4bit`/`fedavg`/`α=0.1`)와
run_name이 같아서, 자연 수렴했다면 재실행 없이 그대로 core 결과로
재사용됩니다. 이 개발 환경은 GPU가 없어(`torch.cuda.is_available() ==
False`) 파일럿 자체는 실제 GPU 인스턴스에서 실행해야 합니다.

`compute_compression_alpha_trend` originally assumed a lower-is-better
metric (PPL) when deciding the penalty's sign, so feeding ROUGE-L
(higher-is-better) in directly flipped the "compression hurt performance"
meaning. Three additions fix this:

- Added `higher_is_better=False` (default, backward compatible) — a
  higher-is-better metric like ROUGE-L can be passed in as-is with
  `higher_is_better=True`, no sign conversion needed.
- Added `compression="qlora_4bit"` (default) — calling the same function
  again with `"qlora_8bit"` lets the two correlation coefficients be
  compared to check "is 4-bit more sensitive to non-IID than 8-bit"
  (a dose-response check).
- Added `per_category_compression_penalty()` — breaks the same
  correlation analysis down per task category, screening (via a z-score
  on the relative penalty) which categories are especially vulnerable to
  the compression x non-IID interaction (relative ranking, not a rigorous
  significance test, since there are only 8 categories).

`scripts/analyze_interaction.py` now prints the 4-bit/8-bit trends side by
side, for both PPL and ROUGE-L. `scripts/run_experiment.py` gained a
`--num-rounds` override — meant for piloting the hardest combination
(`qlora_4bit`, `alpha=0.1`) with a generous cap (e.g. 30) before running
the full 18 combinations, to read off its actual `converged_round` and set
`experiment_config.yaml`'s `federated.num_rounds` to that value plus a
margin:

```bash
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 0.1 --num-rounds 30
```

This pilot run shares its run_name with one of the core 18 combinations
(`qlora_4bit`/`fedavg`/`alpha=0.1`), so if it converges naturally there's
no need to rerun it — it's reused as-is for the core results. This dev
environment has no GPU (`torch.cuda.is_available() == False`), so the
pilot itself needs to run on an actual GPU instance.

### Subtask 2.3 카테고리별 취약도 스크리닝 — 이제 실행부까지 배선됨 (Subtask 2.3 Per-Category Vulnerability Screening — Now Wired Up)
`per_category_compression_penalty`는 만들어만 두고 호출부가 없어서 실제
데이터를 넣어본 적이 없었습니다. `score_generations_by_category`(신규 —
`*_generations.jsonl`에서 예제별 ROUGE-L을 다시 계산해 카테고리별로 묶음)를
추가하고 `scripts/analyze_interaction.py`의 `main()`에 실제로 연결해서,
18개 로그가 갖춰지면 자동으로 카테고리별 취약도(z-score) 스크리닝까지
출력하도록 했습니다. 가짜 18조합 로그로 end-to-end 검증까지 마쳤습니다
(의도적으로 취약하게 설계한 카테고리가 정확히 플래그됨).

`per_category_compression_penalty` was written but had no caller — it was
never actually fed real data. Added `score_generations_by_category`
(new — recomputes per-example ROUGE-L from `*_generations.jsonl` and
groups it by category) and wired it into
`scripts/analyze_interaction.py`'s `main()`, so once all 18 logs exist the
per-category vulnerability (z-score) screening prints automatically.
Verified end-to-end against synthetic 18-combination logs (a
deliberately-designed vulnerable category was correctly flagged).

---

## 실행 방법 (How to Run)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**테스트 결과**: **63 passed, 2 skipped**(QLoRA 4bit/8bit, GPU 필요) — 전체 스위트.
`test_data.py`/`test_metrics.py`/`test_scaffold.py`가 추가돼(그동안 어떤
테스트에서도 직접 호출되지 않던 `src/data.py`·`src/metrics.py`·
`src/scaffold.py` 함수들의 커버리지 공백을 메꿈), `fl_runner.py`가 그동안
버리고 있던 `peak_vram_gb`/`total_latency_sec` 로깅도 고쳤습니다.
`score_generations_by_category` 테스트 2개도 추가됐습니다(위 참고).

**Test results**: **63 passed, 2 skipped** (QLoRA 4-bit/8-bit, requires GPU) — full suite.
Added `test_data.py`/`test_metrics.py`/`test_scaffold.py` (closing a
coverage gap for `src/data.py`/`src/metrics.py`/`src/scaffold.py`
functions that no test had ever called directly), and fixed
`fl_runner.py`, which was discarding `peak_vram_gb`/`total_latency_sec`
instead of logging them. Also added 2 tests for
`score_generations_by_category` (see above).

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
