# Dolly-15k FL × PEFT Research Placement — Week 5 Progress Report

**Research Question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist as PEFT quantization (LoRA→QLoRA) becomes more aggressive?

---

## Progress Summary

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, PEFT wiring | ✅ Done |
| Week 2 | Dolly-15k pipeline, Dirichlet(α=1) partitioning | ✅ Done |
| Week 3 | FedProx/SCAFFOLD integration, checkpoint/convergence criteria | ✅ Done |
| Week 4 | Core 18-combination (3 compression × 2 FL × 3 alpha) run script + compression×non-IID correlation analysis functions | ✅ Done |
| **Week 5** | Redesign per advisor feedback + correlation/selection logic tests complete | ✅ Done (this branch) |
| Week 6 | Final report | Planned (`week6`) |

---

## Week 1–4 Summary

- **Week 1**: PEFT wiring (LoRA/QLoRA/DoRA)
- **Week 2**: Dolly-15k pipeline, Dirichlet(α=1) partitioning
- **Week 3**: FedProx/SCAFFOLD, checkpoints, convergence criteria, FL integration tests
- **Week 4**: `configs/experiment_config.yaml`, `src/evaluate.py` (compression×alpha correlation),
  `scripts/run_experiment.py`, `scripts/analyze_interaction.py`

## What Was Done in Week 5

### Redesign per Advisor Feedback

Following advisor feedback that "this project's focus should be the
correlation between compression rate and non-IID intensity," the core
design was revised from 3 FL × 2 PEFT (6 combinations) to **3
compression levels (LoRA/QLoRA-8bit/QLoRA-4bit) × 2 FL algorithms
(FedAvg/FedProx) × 3 Dirichlet alphas (0.1/1/10) = 18 combinations**.
SCAFFOLD is dropped from the core design, though `src/scaffold.py` and
its tests are kept as-is. The model was also downsized further, from
Qwen2.5-3B through Qwen2.5-1.5B to **Qwen2.5-0.5B-Instruct** (to cut GPU
time/cost even more -- unlike 1.5B, 0.5B is an exact size that actually
exists in the Qwen2.5 lineup). See the "Design Note" section on the
`week4` branch for the full background.

### `tests/test_evaluate.py` extended

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

### Hardening `compute_compression_alpha_trend`: ROUGE-L Sign, qlora_8bit Comparison, Per-Category Vulnerability

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

### Subtask 2.3 Per-Category Vulnerability Screening — Now Wired Up

`per_category_compression_penalty` was written but had no caller — it was
never actually fed real data. Added `score_generations_by_category`
(new — recomputes per-example ROUGE-L from `*_generations.jsonl` and
groups it by category) and wired it into
`scripts/analyze_interaction.py`'s `main()`, so once all 18 logs exist the
per-category vulnerability (z-score) screening prints automatically.
Verified end-to-end against synthetic 18-combination logs (a
deliberately-designed vulnerable category was correctly flagged).

### Evaluation Is Now Performed by the Server Directly, per Advisor Feedback

Reflects advisor feedback ("evaluation is based on the test on server,
not individual clients"). Previously, `fl_runner.py` (the server loop)
performed PPL/ROUGE-L evaluation via `clients[0].evaluate()` — Flower's
`NumPyClient` interface (a client-side method). Since the data itself is
the globally-shared held-out set, the result was numerically identical to
a true server-side evaluation, but architecturally it read as "a client
evaluating itself."

Added `src/server_eval.py` (new), with plain functions
(`evaluate_global_model`, `evaluate_global_model_generation`) that take
only a model and a dataset, and changed `fl_runner.py` to call them
directly. `FlowerClient.evaluate()` is kept for Flower-interface
compatibility but now just delegates to the same functions internally —
the actual server loop no longer calls that method. Taken a step further,
evaluation no longer reuses `clients[0].model` — `run_federated_training()`
now creates a **dedicated server-only model instance (`server_model`)**
that belongs to no client. This is still a local simulation sharing the
same process/GPU (8+1=9 model instances now coexist), but the code no
longer reads as "client 0 is doing the evaluating." Added direct unit
tests in `tests/test_server_eval.py`.

---

## How to Run

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**Test results**: **65 passed, 2 skipped** (QLoRA 4-bit/8-bit, requires GPU) — full suite.
Added `test_data.py`/`test_metrics.py`/`test_scaffold.py` (closing a
coverage gap for `src/data.py`/`src/metrics.py`/`src/scaffold.py`
functions that no test had ever called directly), and fixed
`fl_runner.py`, which was discarding `peak_vram_gb`/`total_latency_sec`
instead of logging them. Also added 2 tests for
`score_generations_by_category`, and 2 for the server-side evaluation
functions (`test_server_eval.py`, see above).

---

## Next Week (Week 6) Preview

- Write final report: research question/redesign background/results interpretation/
  limitations (SCAFFOLD dropped from core, single-seed, shared held-out)
- Prepare presentation

→ See the `week6` branch (code identical to this branch; only the README is
replaced with the final report).
