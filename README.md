# Dolly-15k FL × PEFT Research Placement — Week 2 Progress Report

**Research Question**: Do the benefits of FL drift correction (FedAvg→FedProx→SCAFFOLD) persist as PEFT quantization (LoRA→QLoRA) becomes more aggressive?

> This README reflects the **cumulative Week 1–2** status.

---

## Progress Summary

| Week | Goal | Status |
|---|---|---|
| Week 1 | Verify model+GPU environment, wire up PEFT (LoRA/QLoRA/DoRA), reduce loss on a single client | ✅ Done |
| **Week 2** | Dolly-15k pipeline, parameter round-trip verification, Dirichlet(α=0.5) partitioning | ✅ Done (this branch) |
| Week 3 | Integrate FedProx/SCAFFOLD + checkpoint/convergence criteria | Planned (`week3`) |
| Week 4 | Run Core 6 combinations + DoRA/local-epoch diagnostics | Planned |
| Week 5 | Interaction analysis + per-category/convergence-trajectory diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Week 1 Summary (Last Week)
- `src/models.py`: Wired up LoRA (r=8) / QLoRA (NF4) / DoRA (unquantized diagnostic control)
- `src/communication.py`: FL parameter extraction helpers
- `tests/test_model_wiring.py`, `scripts/run_dev_pilot.py`
- Swapped the dev model from `gpt2` → `tiny-random-Llama` (after discovering and fixing an attention-structure mismatch)

---

## What Was Done in Week 2

### 1. Dolly-15k Data Pipeline (`src/data.py`)
- `load_raw_dolly15k()`: Loads the raw HuggingFace `databricks/databricks-dolly-15k` dataset (15,011 examples).
- `build_holdout_split(holdout_fraction=0.1)`: Secures a shared held-out evaluation set by pulling 10% **evenly per category** (performed once before any training starts, fixed independently of client distribution).
- `format_prompt()` / `tokenize_example()`: Applies the `### Instruction / ### Context / ### Response` template; prompt spans are masked with `labels=-100` so that only the response contributes to loss.
- `TokenizedDolly` Dataset, `build_client_dataloaders()`, `build_category_tagged_holdout()`

### 2. Dirichlet Non-IID Partitioning (`src/partitioning.py`)
- `partition_by_category(alpha=0.5, min_category_threshold=150)`: Distributes data non-IID across 8 clients using Dirichlet(α) keyed on category. Categories with fewer samples than the threshold (150) are upsampled via resampling.
- `heterogeneity_score()`: Utility to quantitatively verify that a partition is actually non-IID (score should increase as α decreases).

### 3. Verification
- `tests/test_roundtrip.py`: Verifies that `state_dict_to_ndarrays` ↔ `ndarrays_to_state_dict` preserves both values and key order — if this breaks, the server-client parameter exchange itself is broken.
- `tests/test_partitioning.py`: Verifies that resampling actually fills the threshold, and that non-IID strength increases in the order α=0.1 > 0.5 > 1.0.

### Design Notes
- The held-out set is **shared globally** (not split per client) — see the Week 6 limitations section for the rationale.
- α=0.5 is a **fixed value**. The original draft plan included an α sweep (1.0/0.5/0.1), but it was dropped within the 6-week scope since it is not directly required for the core question (correction × quantization interaction). **(Week 5 update)** This α sweep was reinstated into the core design following advisor feedback — see the `week5`/`week6` branches for details.

---

## How to Run

```bash
pytest tests/test_model_wiring.py tests/test_roundtrip.py tests/test_partitioning.py -v