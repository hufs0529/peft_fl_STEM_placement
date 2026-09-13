# Dolly-15k FL × PEFT Research Placement

A 6-week research placement project codebase. It implements the research
proposal *"Does Drift-Correction Sophistication Interact with PEFT
Quantisation Noise in Federated LLM Fine-Tuning?"* directly in code.

---

## 0. The Question This Project Asks — Research Proposal Aims Mapped to Code

| Aim | Question summary | Code that produces the answer |
|---|---|---|
| **Aim 1** (Task 1) | Does the benefit of drift-correction sophistication (FedAvg→FedProx) hold consistently as PEFT compression intensifies (uncompressed→8-bit→4-bit)? Is this benefit correlated with non-IID intensity (Dirichlet alpha)? Verified with a 3 compression × 2 FL × 3 alpha factorial design. | `src/fl_client.py` (FedProx proximal term), `src/evaluate.py::compute_compression_alpha_trend` (compression×alpha correlation) |
| **Aim 2** (Task 2) | Given the observed correlation, is the best-performing combination robust to a larger local_epochs? | `src/evaluate.py::select_best_performing_combination`, `src/evaluate.py::per_category_breakdown` |

**Core design principle**: This codebase fills the "compression intensity ×
non-IID intensity" grid directly with 18 core combinations (3 compression ×
2 FL × 3 alpha), and Task 2 is limited to exactly the one diagnostic
experiment (local-epoch robustness check) needed to explain **why** the
result shown by that grid appears — unrelated axes (Prompt Tuning, partial
participation, an LLM-judge, centralised baselines) are out of scope for
this research proposal and are not included.

---

## 1. Folder Structure

```
dolly15k-fl-peft-v2/
├── README.md                     # this file
├── requirements.txt
├── configs/
│   ├── dev_config.yaml           # for Week 1-3 verification (gpt2, 2 clients, 2 rounds)
│   └── experiment_config.yaml    # for the Week 4 main experiment (Qwen2.5-0.5B, 8 clients, 10 rounds; revised in Week 5)
├── src/                          # core logic shared by dev/experiment
│   ├── models.py                   # wraps LoRA/QLoRA (4-bit/8-bit, core) and DoRA (unused diagnostic control)
│   ├── data.py                     # Dolly-15k loading, tokenisation, held-out split
│   ├── partitioning.py             # Dirichlet (alpha=1) non-IID + category resampling
│   ├── communication.py            # FL parameter round-trip + payload size computation
│   ├── fl_client.py                # shared client (FedProx proximal term, warmup, VRAM/latency, ROUGE-L)
│   ├── fl_runner.py                # manual round loop — wires up convergence, checkpointing, logging, W&B
│   ├── server_eval.py              # server-side global-model evaluation (PPL/ROUGE-L) — bypasses NumPyClient.evaluate()
│   ├── scaffold.py                 # actual SCAFFOLD implementation (control variate) — dropped from core, code/tests kept
│   ├── convergence.py              # early stop on 3 consecutive rounds of <1% improvement
│   ├── checkpointing.py            # per-round save/resume
│   ├── metrics.py                  # VRAM, latency, ROUGE-L, trainable param count
│   └── evaluate.py                 # per-category breakdown, client fairness, compression x alpha correlation
├── tests/                        # see the "Test Composition" section below
├── scripts/
│   ├── run_dev_pilot.py            # Week 1: single-client verification with the real model
│   ├── run_experiment.py           # entry point for Task 1's core 18 combinations (3 compression x 2 FL x 3 alpha) + Task 2 diagnostics
│   ├── analyze_interaction.py      # Subtask 1.3/2.2/2.3: compression x alpha correlation + per-category vulnerability screening + diagnostic target selection
│   └── plot_results.py             # visualizes the 18 combinations as PPL/ROUGE-L/communication/convergence/VRAM/latency figures + a summary table
└── results/
    ├── checkpoints/{run_name}/round_NNN.pt   # per-round checkpoints (excluded from git tracking)
    ├── logs/{run_name}_rounds.jsonl          # full per-round metrics
    ├── logs/{run_name}_generations.jsonl     # predictions/references from the final generation round (Subtask 2.3 input)
    └── figures/                              # plot_results.py output (6 figures + summary_table.md)
```

---

## 2. How the Task 1 Core Design Is Reflected in the Code

The revised core design (**3 compression × 2 FL × 3 alpha = 18
combinations**) is compressed into a single script (`run_experiment.py`)
with three axes of arguments: `--peft` (+ `--qlora-bits`), `--fl`, and
`--alpha`. In other words, the 18 commands below make up the entire core:

```bash
for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    for alpha in 0.1 1 10; do
      python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
    done
  done
done
```

Once all 18 have finished:

```bash
python scripts/analyze_interaction.py
```

is run. This script computes four things:

1. **Compression×alpha grid** — prints the val_perplexity of all 18
   combinations as a (compression, FL, alpha) grid.
2. **Subtask 1.3 compression×non-IID correlation** — at each (FL, alpha)
   point, computes the "compression penalty" (QLoRA-4bit/8bit performance
   minus LoRA/uncompressed performance), and its Pearson correlation with
   alpha to check whether the penalty grows as alpha decreases (non-IID
   intensifies). **4-bit and 8-bit are computed separately** and compared
   side by side (a more negative correlation for 4-bit than 8-bit is a
   dose-response signal), and the same computation is **repeated for
   ROUGE-L** (higher-is-better, `higher_is_better=True`) in addition to
   PPL, to cross-check whether the two performance metrics point the same
   direction. This is the direct quantitative answer to the advisor's
   feedback ("look at the correlation between compression rate and
   non-IID intensity") (`src/evaluate.py::compute_compression_alpha_trend`).
3. **Subtask 2.3 per-category vulnerability screening** — recomputes
   per-example ROUGE-L from `*_generations.jsonl`
   (`score_generations_by_category`), groups it by category, and screens
   (via z-score) which task categories are especially vulnerable to the
   compression × non-IID interaction
   (`src/evaluate.py::per_category_compression_penalty`).
4. **Subtask 2.2 target selection** — picks the combination among the 18
   with the best task performance as the target for the local_epochs=5
   robustness check.

Once all 18 logs exist, `python scripts/plot_results.py` visualizes the
same data as 6 figures (PPL/ROUGE-L/communication/convergence-speed/
**VRAM/total training latency**) plus a summary table
(`results/figures/`).

### Deriving the num_rounds Ceiling via a Pilot Run

`experiment_config.yaml`'s `federated.num_rounds` (currently 10) needs to
be generous enough that the convergence criterion (3 consecutive rounds of
<1% improvement) actually fires rather than being cut off by an arbitrary
cap. It's recommended to first pilot the combination expected to converge
slowest among the 18 (`qlora --qlora-bits 4 --alpha 0.1` — the heaviest
compression × strongest non-IID) with a generous cap, read off its actual
`converged_round`, and derive the ceiling from that:

```bash
# 1) Run one pilot with a generous cap (before the full 18-combination run)
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 0.1 --num-rounds 30

# 2) Read "converged_round" off the output, and set experiment_config.yaml's
#    federated.num_rounds to that value plus a margin (e.g. +3-5)
# 3) This pilot run shares its run_name with one of the core 18 combinations
#    (qlora_4bit/fedavg/alpha=0.1), so if it converged naturally there's no
#    need to rerun it — it's reused as-is for the core results.
```

`--num-rounds` is meant only for this one-off pilot override — the full 18
runs can just use the value baked into `experiment_config.yaml`, no need to
pass it every time. After all 18 are done, make sure none of them have a
`converged_round` of `None` (never converged within budget) — if one does,
that combination was truncated by the round cap, which is indistinguishable
from a genuinely worse compression penalty in the compression×alpha
correlation analysis.

> This repo's local development environment (no GPU,
> `torch.cuda.is_available() == False`) cannot actually run a QLoRA pilot
> — the procedure above needs to run on the GPU instance described in
> section 6.

### Task 2 Diagnostic Execution

| What it verifies | Command |
|---|---|
| Local epoch 1 vs 5 robustness check (applied to the best-performing combination) | `run_experiment.py --peft <best_peft> --fl <best_fl> --alpha <best_alpha> [--qlora-bits 8] --local-epochs 5` |

`<best_peft>` / `<best_fl>` / `<best_alpha>` can simply be copied from the
output of `analyze_interaction.py`.

Subtask 2.3 (per-category breakdown) is computed directly from
`per_category_breakdown` in `src/evaluate.py` and the per-round metrics
already logged in `results/logs/*_rounds.jsonl`, with no additional GPU
cost.

---

## 3. How Each Module Maps to the Research Proposal

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

### Dropped from Core — Code/Tests Kept

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
- The `(mu/2)*||local-global||^2` proximal term is added to the loss only
  under FedProx (`is_fedprox` branch, μ=0.01).
- Linear warmup is applied only in the first round (`server_round==1`).
- Wrapped in `track_vram_and_latency()` to automatically measure Peak
  VRAM/Training Latency.
- `evaluate()` is kept only to satisfy Flower's `NumPyClient` interface —
  per advisor feedback ("evaluation is based on the test on server, not
  individual clients"), the actual server loop (`fl_runner.py`) never
  calls it, calling `src/server_eval.py`'s pure functions directly
  instead. `evaluate()` internally reuses those same functions, so there's
  no duplicated logic (see the `fl_runner.py` section below).

### `src/fl_runner.py` — Subtask 1.2, 2.3
- **Convergence criterion**: `ConvergenceTracker` takes the validation loss
  every round and stops the loop after 3 consecutive rounds of <1%
  improvement (≤10 rounds).
- **Checkpointing**: saved every round, and automatically resumed from the
  last checkpoint at start-up (`resume_or_start_fresh`) — safe even if a
  session is interrupted.
- **Server-side evaluation** (advisor feedback: "evaluation is based on
  the test on server, not individual clients"): the server loop calls
  `src/server_eval.py`'s `evaluate_global_model`/
  `evaluate_global_model_generation` — plain functions taking only a model
  and a dataset — directly, rather than going through Flower's
  `NumPyClient.evaluate()` (a client-side interface). Evaluation no
  longer borrows `clients[0].model` — a **dedicated server-only model
  instance (`server_model`)**, belonging to no client, is created
  separately. This is still a local simulation sharing the same
  process/GPU (8+1=9 model instances coexist), but the code no longer
  reads as "client 0 is doing the evaluating." At evaluation time,
  `server_model` holds `global_state` (that round's aggregated `fit()`
  results). Right after aggregation, every client
  is overwritten with the same global parameters and sees the same shared
  held-out set, so evaluating it any number of times gives the same result
  — hence the server evaluates this held-out set exactly once (evaluating
  all 8 would give the same result, so the rest would be pure duplicate
  computation). ROUGE-L (which needs generation) is also computed by the
  server directly, **only once, after the loop ends, on the final
  global_state**, against a category-stratified sample
  (`config['data']['rouge_l_sample_size']`), rather than every round.
- **Logging**: every round's `round_record` (appended as one line to
  `results/logs/{run_name}_rounds.jsonl`) carries `round`, `val_loss`,
  `val_perplexity`, `round_latency_sec`, `communication_bytes_this_round`,
  and **`peak_vram_gb`** (averaged over that round's participating
  clients — FedAvg/FedProx only, already measured by `fit()` in
  `fl_client.py` via `track_vram_and_latency()`). The last round also gets
  `rouge_l`, and generations are saved separately to
  `results/logs/{run_name}_generations.jsonl`. Once the loop ends,
  `run_federated_training()`'s return value (used by the caller as the
  final summary) adds `total_communication_bytes`, **`total_latency_sec`**
  (the sum of each round's `round_latency_sec`), and
  **`avg_peak_vram_gb`** (the average of each round's `peak_vram_gb`) —
  so every metric needed for the compression×non-IID trade-off analysis
  (communication/time/memory) ends up in these two places (the
  `*_rounds.jsonl` file and the return value). SCAFFOLD's
  `scaffold_client_fit()` doesn't perform this measurement, so its
  `peak_vram_gb` stays `None` (intentional, since it's excluded from the
  core analysis).
- **W&B**: if `config['logging']['use_wandb']=true`, the same metrics are
  also pushed to the W&B dashboard in real time.

### `src/evaluate.py` — Subtask 1.3, 2.2, 2.3
- `compute_compression_alpha_trend`: the function that computes **the
  direct answer to the advisor's feedback**. From the performance values
  of the 18 combinations, it derives the "compression penalty" (QLoRA-4bit
  or 8-bit, picked via the `compression` argument, minus LoRA) at each
  (FL, alpha) point, and its Pearson correlation with alpha. It is
  computed for both `val_perplexity` (performance) and `rounds_run`
  (convergence speed, a value already logged for free) via
  `performance_field`, so the two metrics can be cross-checked for whether
  they point the same direction (`scripts/analyze_interaction.py`). A
  higher-is-better metric like ROUGE-L can be passed in as-is with
  `higher_is_better=True` (omitting it flips the penalty's sign). Returns
  0.0 instead of NaN when the penalty is independent of alpha (zero
  variance).
- `per_category_compression_penalty`: a per-category breakdown of the
  correlation analysis above. Computes each task category's mean relative
  penalty (a ratio, `(lora-compression)/lora`, rather than a raw
  difference, to avoid categories' differing absolute PPL scales), and a
  z-score across categories (`vulnerable=True` above the default threshold
  of 1.0), to screen which categories are especially vulnerable to the
  compression × non-IID interaction. With only a handful of categories
  (8 in Dolly), this is relative-ranking screening, not a significance
  test — categories with a small sample size (`n`) are flagged separately.
  `scripts/analyze_interaction.py` actually calls this, fed by
  `score_generations_by_category`.
- `score_generations_by_category`: a helper that recomputes per-example
  ROUGE-L from `*_generations.jsonl` (predictions/references/categories)
  and groups it by category, building the `{category: {"rouge_l":..,
  "n":..}}` input `per_category_compression_penalty` needs. Only supports
  ROUGE-L, since PPL isn't logged per example.
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

## 4. Workflow — Mapped to the 6-Week Timeline

| Week | Command run | What it verifies |
|---|---|---|
| Week 1 | `pytest tests/test_model_wiring.py` → `python scripts/run_dev_pilot.py --stage single_client` | model+GPU environment check, PEFT wiring (LoRA/QLoRA/DoRA), single-client loss decrease |
| Week 2 | `pytest tests/test_roundtrip.py tests/test_partitioning.py` | Dolly-15k pipeline, parameter round-trip, Dirichlet (α=1) partitioning verification |
| Week 3 | `pytest tests/test_fl_integration.py tests/test_convergence.py tests/test_checkpointing.py` | FedProx/SCAFFOLD integration, checkpointing/convergence criterion, small-scale rehearsal of the 6 combinations |
| Week 4 | `run_experiment.py` × 18 (core) → `analyze_interaction.py` → local-epoch robustness check | Subtask 1.2 main experiment (revised design) + Subtask 2.2 diagnostic |
| Week 5 | `pytest tests/test_evaluate.py` → calling `src/evaluate.py` functions directly from an analysis notebook | Subtask 1.3 compression×non-IID correlation analysis, Subtask 2.3 per-category diagnostics |
| Week 6 | write the report (noting the redesign background, SCAFFOLD dropped from core, and single-seed limitations) + present | — |

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## 5. Test Composition

| File | What it verifies | Corresponding week |
|---|---|---|
| `test_model_wiring.py` | LoRA/QLoRA (4-bit/8-bit)/DoRA wiring; that an undefined PEFT type raises `ValueError` | Week 1, 2 (8-bit added) |
| `test_roundtrip.py` | value/order preservation when FL parameters are sent and received | Week 2 |
| `test_partitioning.py` | minimum-category-threshold resampling; that a smaller α actually produces a more non-IID split | Week 2 |
| `test_data.py` | prompt formatting, **label masking** (that -100 covers exactly the prompt span), even category split for the held-out set, category-stratified sampling for ROUGE-L, client DataLoader construction (only `load_raw_dolly15k` is network-marked) | Week 2 |
| `test_fl_integration.py` | all three of FedAvg/FedProx/SCAFFOLD complete the full loop; SCAFFOLD's `local_control` is preserved across rounds; FedProx's `fit()` runs correctly; **`peak_vram_gb`/`total_latency_sec` aggregation**; **resuming a checkpoint already at `num_rounds` raises a clear `RuntimeError` instead of crashing** | Week 3 |
| `test_convergence.py` | actually stops after 3 consecutive rounds of <1% improvement; does not stop while improvements are large | Week 3 |
| `test_checkpointing.py` | round/state are restored exactly after save-then-resume | Week 3 |
| `test_metrics.py` | ROUGE-L computation (identical/unrelated strings, averaging), trainable parameter counting, the VRAM/latency measurement context manager, response generation | Week 3 |
| `test_scaffold.py` | that the control-variate aggregation formulas match hand-computed values (delta_y averaging, global_control scaling by participation rate); that one local training step actually updates parameters | Week 3 |
| `test_server_eval.py` | that `evaluate_global_model`/`evaluate_global_model_generation` correctly compute PPL/ROUGE-L without going through `NumPyClient.evaluate()` (advisor feedback: server-side evaluation) | Week 3 |
| `test_evaluate.py` | per-category breakdown, fairness variance, **compression×alpha correlation** (penalty values, negative correlation, 0.0 for constant penalty, qlora_8bit comparison, ROUGE-L `higher_is_better` sign), **per-category vulnerability screening** (`per_category_compression_penalty`, incl. its input contract with `score_generations_by_category`), **selecting the best-performing combination**, (legacy) interaction-effect computation | Week 4-6 |

---

## 6. Execution Environment

- **Week 1-3 (development/debugging)**: Google Colab Pro — the base
  subscription is enough since the work is mostly miniature verification
  with low compute-unit consumption.
- **Week 4 (main experiment)**: an on-demand or spot single-GPU instance
  (AWS g5.xlarge or Lambda Labs A100) — since `src/checkpointing.py` saves
  every round, it is safe to resume even if the spot instance is
  reclaimed. Paid Colab tiers (Pro/Pro+) are not recommended for the main
  experiment's scale (below) — even Colab Pro+'s monthly compute-unit
  budget is roughly 35-40 hours of A100 time, less than what's needed.
  (Week 5 update: after downsizing the model further, from Qwen2.5-1.5B to
  Qwen2.5-0.5B, the core 18 runs plus 1 diagnostic run — 19 total — are
  estimated at roughly 35-48 GPU-hr on a g5.xlarge spot instance, costing
  roughly $12-28 (a linear-scaling approximation from the 1.5B figure of
  105-143 GPU-hr / $35-84, to be re-confirmed with a pilot run) — see the
  design note on the `week4` branch.)

```bash
pip install -r requirements.txt
# bitsandbytes only installs/works correctly in a GPU environment.
```

---

## 7. Known Limitations — To Be Reflected in the Week 6 Report's Limitations Section

- **SCAFFOLD is dropped from the core 18 combinations** (Week 5 advisor
  feedback) — its code (`src/scaffold.py`) and tests remain, but the main
  experiment compares only FedAvg and FedProx. Whether the same
  compression×non-IID correlation holds under a more sophisticated
  correction algorithm like SCAFFOLD is not answered within this scope.
  SCAFFOLD itself also remains a **simplified Option II approximation**.
- The model was **downsized further, from Qwen2.5-3B through Qwen2.5-1.5B
  to Qwen2.5-0.5B-Instruct** (to cut GPU time/cost even more) — whether
  the same correlation pattern holds on a larger model was not verified,
  and 0.5B's lower representational capacity than 3B/1.5B carries some
  risk that the compression x non-IID interaction itself appears weaker
  or different at this scale.
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
  a Pearson correlation computed from only **3 alpha points (0.1/1/10)**
  — with so few points, a non-linear pattern (e.g., a U-shape where the
  penalty peaks at a middle alpha) could be missed.
- `per_category_compression_penalty`'s per-category vulnerability z-scores
  are computed over only **8 Dolly categories**, and stratifying
  `rouge_l_sample_size` (200 by default) across them leaves only ~25
  examples per category — this should be read as **relative screening**,
  not a rigorous significance test, and categories with a particularly
  small sample (`n` in the return value) may have rankings that shift
  across seeds.
- The entire codebase has **only passed syntax verification and has not
  yet been run in an actual GPU environment.** Start by running
  `pytest tests/` on Week 1's first day to confirm real behaviour.
