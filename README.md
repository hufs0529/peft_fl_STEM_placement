# Dolly-15k FL × PEFT Research Placement — Week 1 Progress Report

**Research Question**: Do the benefits of FL drift correction (FedAvg → FedProx → SCAFFOLD) persist even as PEFT quantization noise (LoRA → QLoRA) increases? (`Does Drift-Correction Sophistication Interact with PEFT Quantisation Noise in Federated LLM Fine-Tuning?`)

> This README reflects **only what has been completed up to Week 1**. As the project progresses into subsequent weekly branches (`week2`, `week3`, ...), this document will be cumulatively updated.

---

## Progress Overview

| Week | Target | Status |
|---|---|---|
| **Week 1** | Model + GPU environment setup, PEFT wiring (LoRA/QLoRA/DoRA), single-client loss reduction | ✅ Completed (this branch) |
| Week 2 | Dolly-15k pipeline + Dirichlet partitioning verification | Planned (`week2` branch) |
| Week 3 | FedProx/SCAFFOLD integration + Checkpointing/Convergence criteria | Planned |
| Week 4 | Core 6-combination execution + DoRA/local-epoch diagnostics | Planned |
| Week 5 | Interaction analysis + Category-wise / Convergence trajectory diagnostics | Planned |
| Week 6 | Final report | Planned |

---

## Work Completed in Week 1

### 1. Project Skeleton
- `requirements.txt`: torch, transformers, peft, bitsandbytes, flwr[simulation], etc.
- `configs/dev_config.yaml`: Miniature configurations for validation in Weeks 1–3.

### 2. PEFT Wiring (`src/models.py`)
Integrated the three PEFT methods required for Research Proposal Subtask 1.1 (Core: LoRA/QLoRA) and Subtask 2.1 (Diagnostics: DoRA) into a unified `get_model(config)` function:

- **LoRA**: r=8, α=16, dropout=0.05, `target_modules=["q_proj","v_proj"]`
- **QLoRA**: Same LoRA configuration as above + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- **DoRA**: Diagnostic control group with equivalent compression to LoRA but without quantization (`use_dora=True`) — used in Task 2 to differentiate "whether interaction stems from quantization or compression itself."
- Raises a `ValueError` for undefined PEFT types.

### 3. FL Parameter Helper (`src/communication.py`)
While the "roundtrip" communication itself is in the Week 2 scope, `get_trainable_state_dict()`—which extracts trainable parameters only—was required for this week's wiring test and has been implemented early.

### 4. Verification
- `tests/test_model_wiring.py`: Verified that trainable parameters are correctly captured for LoRA/DoRA, forward passes complete without errors, and undefined PEFT types trigger a `ValueError` (QLoRA is tested in GPU environments only due to `bitsandbytes` 4-bit constraints).
- `scripts/run_dev_pilot.py`: Verified that running 5 training steps on a single client with the actual model results in a measurable loss decrease.

### Design Note: Dev Model Selection
Initially, `gpt2` was considered as the development model. However, GPT-2's attention uses a fused `c_attn` Conv1D layer, which is incompatible with `target_modules=["q_proj","v_proj"]` and causes LoRA injection to fail. This was resolved by switching to `hf-internal-testing/tiny-random-LlamaForCausalLM`, which shares the **exact `q_proj`/`v_proj` architecture** as the primary experimental model (Qwen2.5-3B). Using a miniature model with matching architecture ensures that Week 1's wiring validation remains functionally meaningful.

---

## How to Run

```bash
pip install -r requirements.txt
pytest tests/test_model_wiring.py -v
python scripts/run_dev_pilot.py --stage single_client