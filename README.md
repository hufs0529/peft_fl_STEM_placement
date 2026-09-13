# Dolly-15k FL × PEFT Research Placement — Week 3 Progress Report

**Research Question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD)
persist as PEFT quantization intensifies (LoRA→QLoRA)?

> This README reflects the **cumulative state of Weeks 1–3**.

---

## Progress Summary

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model + GPU environment, wire up PEFT (LoRA/QLoRA/DoRA) | ✅ Done |
| Week 2 | Dolly-15k pipeline, parameter round-trip, Dirichlet (α=1) partitioning | ✅ Done |
| **Week 3** | FedProx/SCAFFOLD integration, checkpoint/convergence criteria, FL integration tests | ✅ Done (this branch) |
| Week 4 | Run the core experiment (3 compression × 2 FL × 3 alpha = 18 combinations) + local-epoch diagnostic | Planned (`week4`) |
| Week 5 | Compression×non-IID correlation analysis + per-category diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Week 1–2 Summary

- **Week 1**: `src/models.py` (LoRA/QLoRA/DoRA wiring), `src/communication.py`
  (parameter helpers), wiring tests, dev pilot script
- **Week 2**: `src/data.py` (Dolly-15k loading/tokenization/held-out),
  `src/partitioning.py` (Dirichlet α=1 non-IID split), roundtrip/partitioning
  tests

## What Was Done in Week 3 — This Is the Biggest Chunk So Far

### 1. Training/evaluation metric utilities (`src/metrics.py`)
`track_vram_and_latency()` (a context manager wrapping the training loop to
measure Peak VRAM and Training Latency), `compute_rouge_l()`,
`generate_responses()`, `count_trainable_parameters()` — used directly by the
client implementation that follows.

### 2. Convergence criterion (`src/convergence.py`)
`ConvergenceTracker`: judged converged if the improvement rate is below 1%
for 3 consecutive recent rounds. The logic was validated independently
before wiring it into the FL loop.

### 3. Checkpointing (`src/checkpointing.py`)
Saves every round (`save_checkpoint`), automatically resumes on restart
(`resume_or_start_fresh`) — handles spot-instance reclamation and dropped
sessions.

### 4. FL client (`src/fl_client.py`)
- Adds the `(μ/2)·‖local-global‖²` proximal term to the loss only for
  FedProx (μ=0.01)
- Linear warmup only on the first round
- `evaluate()` always computes PPL (teacher-forcing forward pass), while
  ROUGE-L (requires generation, expensive) is computed only on the separate,
  smaller `rouge_eval_dataset`, and only when explicitly requested

### 5. SCAFFOLD (`src/scaffold.py`)
A **simplified approximation** of Karimireddy et al. (2020) Option II.
Consists of adding the control variate correction directly to the gradient
(`scaffold_client_fit`), and aggregation that jointly updates the global
model and the global control variate (`scaffold_aggregate`) — a separate
path from FedAvg/FedProx's weighted-average aggregation.

### 6. Manual FL round loop (`src/fl_runner.py`)
`flwr.simulation.start_simulation()` makes it hard to stop early or resume
mid-loop, so we use a loop that directly implements aggregation, evaluation,
checkpointing, and convergence judgment (see the diagram above:
fit → aggregate → PPL evaluation → checkpoint → convergence check →
early-stop or next round).

ROUGE-L generation is computed **exactly once, with the final parameters,
after the loop ends** — not every round — because autoregressive generation
is far more expensive.

### 7. Validation
- `test_convergence.py`, `test_checkpointing.py`: standalone validation of
  each unit
- `test_fl_integration.py`: verifies that **all three paths** — FedAvg,
  FedProx, and SCAFFOLD — pass through the full loop (fit → aggregate →
  evaluate → checkpoint → convergence judgment) without error, that
  SCAFFOLD's `local_control` is preserved across rounds, that FedProx's
  proximal term actually takes effect, and that `peak_vram_gb`/
  `total_latency_sec` are aggregated correctly (see 8 below)

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

### Design note: why is PPL measured with only one client?
Right after aggregation, all 8 clients are overwritten with the **same
global parameters** and see the **same shared held-out set** — running the
same model on the same data 8 times always gives the same result (aside from
floating-point error). So only one representative client is evaluated, and
the other 7 duplicate computations are skipped. (Conversely, `fit()` needs
all 8, since each client has different local data.)

---

## How to Run

```bash
pytest tests/test_convergence.py tests/test_checkpointing.py tests/test_fl_integration.py -v
```

**Test results**: 45 passed, 2 skipped (QLoRA 4-bit/8-bit, requires GPU),
based on `pytest tests/ -m "not network"` — all Week 1–3 tests pass.
`test_data.py` (unit tests for `src/data.py`, incl. label masking),
`test_metrics.py` (ROUGE-L/trainable params/VRAM measurement), and
`test_scaffold.py` (verifying the control-variate aggregation formulas)
were added, closing a coverage gap for functions that no test had ever
called directly.

---

## Preview of Next Week — Week 4

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
