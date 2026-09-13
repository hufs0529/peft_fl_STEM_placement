# Dolly-15k FL × PEFT Research Placement — Week 4 Progress Report

**Research Question**: Does the benefit of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist even as PEFT quantization (LoRA→QLoRA) intensifies?

> This README reflects the **cumulative Week 1-4** status.

---

## Progress Summary

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, PEFT wiring (LoRA/QLoRA/DoRA) | ✅ Done |
| Week 2 | Dolly-15k pipeline, Dirichlet(α=1) partitioning | ✅ Done |
| Week 3 | FedProx/SCAFFOLD integration, checkpoint/convergence criteria | ✅ Done |
| **Week 4** | Core 18-combination (3 compression × 2 FL × 3 alpha) run script + compression×non-IID correlation analysis | ✅ Done (this branch) |
| Week 5 | Correlation analysis tests + per-category diagnostics | Planned (`week5`) |
| Week 6 | Final report | Planned |

---

## Week 1-3 Summary

- **Week 1**: PEFT wiring (LoRA/QLoRA/DoRA), parameter helpers, wiring tests
- **Week 2**: Dolly-15k pipeline, Dirichlet(α=1) partitioning
- **Week 3**: FedProx/SCAFFOLD, checkpointing, convergence criteria, manual FL round loop,
  FL integration tests (18 passed)

## What Was Done in Week 4

### Main Experiment Configuration

Qwen2.5-0.5B-Instruct (downsized further, from 3B through 1.5B, to cut GPU
time/cost even more), 8 clients, up to 10 rounds, effective batch 16.
`peft.qlora_bits` (4/8) and `partitioning.alpha` are overridden via
`--qlora-bits`/`--alpha` in `scripts/run_experiment.py`.
`rouge_l_sample_size` controls the cost of
generation evaluation.

### Compression×Non-IID Correlation — This Project's Key Deliverable

- `compute_compression_alpha_trend(run_results, performance_field, compression="qlora_4bit", higher_is_better=False)`:
  at each (FL, alpha) point, computes the "compression penalty"
  (QLoRA-4bit or 8-bit, picked via `compression`, minus LoRA/uncompressed
  performance) and its Pearson correlation with alpha. Whether that
  penalty grows as alpha decreases (non-IID intensifies) is the direct
  quantitative answer to the advisor's question. Calling it twice with
  different `compression` values compares 4-bit and 8-bit side by side; a
  higher-is-better metric like ROUGE-L can be passed in as-is with
  `higher_is_better=True`.
- `per_category_compression_penalty`: breaks the above correlation down
  per task category, via a z-score, to screen which categories are
  especially vulnerable to the compression x non-IID interaction.
  `scripts/analyze_interaction.py` actually calls it, fed by
  `score_generations_by_category` (recomputes per-example ROUGE-L from
  `*_generations.jsonl`)
- `select_best_performing_combination`: selects the target for Subtask 2.2
  local_epochs=5 (now scoped over all 18 combinations)
- `per_category_breakdown` / `client_fairness_variance` / `total_communication_cost`:
  general-purpose utilities for Subtask 2.3 diagnostics
- `compute_interaction_effects` / `select_largest_interaction_fl_algorithm`
  (the original core-6/SCAFFOLD interaction computation) are kept as-is but
  are no longer called by the new core design — available for a separate
  diagnostic if needed later

### Run Script
```bash
for compression in lora "qlora --qlora-bits 8" "qlora --qlora-bits 4"; do
  for fl in fedavg fedprox; do
    for alpha in 0.1 1 10; do
      python scripts/run_experiment.py --peft $compression --fl $fl --alpha $alpha
    done
  done
done
```

Before running all 18, it's recommended to first pilot the hardest
combination (`qlora --qlora-bits 4 --alpha 0.1`) with `--num-rounds` set
to a generous cap, read off its actual `converged_round`, and set
`experiment_config.yaml`'s `federated.num_rounds` (currently 10) to that
value plus a margin — otherwise only the hardest combination gets
truncated by the round cap, and the compression×alpha correlation
analysis can't distinguish "compression is genuinely worse" from "it just
ran out of rounds":

```bash
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 0.1 --num-rounds 30
```

### Result Analysis Script
```bash
python scripts/analyze_interaction.py
```

Reads the logs of the 18 runs, prints the compression×alpha grid and
correlation, runs the **per-category compression×non-IID vulnerability
screening** (`*_generations.jsonl` → `score_generations_by_category` →
`per_category_compression_penalty`), and automatically reports the
combination to use for the Task 2 diagnostic run (see the code block
above for invocation and sample output).

### Design Note: Redesign Following Advisor Feedback

As of Week 4, the original plan was 8 runs total (the core 6 combinations
[3 FL × 2 PEFT] plus 2 Task 2 diagnostic runs — DoRA and
local_epochs=5). In Week 5, the advisor gave feedback that "this
project's focus should be the correlation between compression rate and
non-IID intensity," so the core design was revised as follows:

- **3 compression levels** (LoRA=uncompressed / QLoRA 8-bit / QLoRA
  4-bit) × **2 FL algorithms** (FedAvg/FedProx — SCAFFOLD is dropped from
  the core design, though its code/tests are kept) × **3 Dirichlet alpha
  levels** (0.1/1/10) = **18 runs**
- The model was downsized from Qwen2.5-3B through Qwen2.5-1.5B to
  **Qwen2.5-0.5B-Instruct** (to cut GPU time/cost further) — staying
  within the Qwen family; unlike 1.5B, 0.5B is an exact size in the
  lineup, not just the closest approximation
- Since `fl_runner.py` (Week 3) already computes PPL once, directly on the
  server, and ROUGE-L once at the last round (also server-side) over a
  category-stratified sample (advisor feedback: evaluate on server, not
  individual clients — `src/server_eval.py`), each run was previously
  estimated at roughly 5.5-7.5 GPU-hr on the 1.5B model; scaling that down by roughly
  the parameter-count ratio (1.5B -> 0.5B, about 1/3) gives roughly
  **1.8-2.5 GPU-hr per run on 0.5B** — we plan for the 18 runs plus 1 Task
  2 diagnostic run (local_epochs=5), 19 runs total, to take roughly
  **35-48 GPU-hr on a g5.xlarge spot instance, costing roughly $12-28**.
  (This is a linear-scaling approximation, not a measurement — the exact
  figure will be derived from a single Week 4 pilot run instead.)

---

## How to Run — Not Yet Executed on a GPU Environment, Only Syntax/Wiring Validated So Far

**Test results**: 48 passed, 2 skipped (QLoRA 4-bit/8-bit, requires GPU —
`test_evaluate.py` will be added in Week 5). Closed a coverage gap for
`src/data.py` (label masking, etc.), `src/metrics.py` (ROUGE-L/VRAM
measurement), and `src/scaffold.py` (control-variate aggregation formulas)
— none of which any test had called directly — via `test_data.py`/
`test_metrics.py`/`test_scaffold.py`. Also fixed `fl_runner.py`, which was
discarding `peak_vram_gb`/`total_latency_sec` instead of logging them into
`round_record`/the final result (needed for the memory axis of
`compute_compression_alpha_trend`). Additionally, per advisor feedback
("evaluate on server, not individual clients"), PPL/ROUGE-L evaluation now
goes through the server calling `src/server_eval.py` directly rather than
`NumPyClient.evaluate()` (`test_server_eval.py` added). Taken a step
further, evaluation no longer reuses `clients[0].model` — it now uses a
**dedicated server-only model instance (`server_model`)** that belongs to
no client. Since this is still a local simulation sharing the same
GPU/process, the underlying hardware usage is unchanged, but the code no
longer reads as "client 0 is doing the evaluating." Also found and fixed
a bug where resuming a checkpoint that had already reached `num_rounds`
crashed with an `IndexError` from an empty `round_records` list — now
raises a clear `RuntimeError` instead, with a regression test added.

---

## Preview of Next Week

- `tests/test_evaluate.py`: unit tests for `compute_compression_alpha_trend`
  (the compression×alpha correlation) plus the existing selection-logic tests
- Run Subtask 2.3 diagnostic analyses such as per-category performance
  breakdown

→ See the `week5` branch.
