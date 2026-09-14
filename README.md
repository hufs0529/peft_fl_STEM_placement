## Licence

This repository is dual-licensed.

| Component | Licence |
|---|---|
| Source code, tests, analysis scripts | Apache-2.0 (`LICENSE`) |
| Experiment logs in `results/logs/` | CC BY-SA 4.0 (`LICENSE-DATA`) |

The logs are share-alike because `*_generations.jsonl` reproduces instructions
and reference responses from databricks-dolly-15k (CC BY-SA 3.0). Neither the
source corpus nor the base model (Qwen2.5-0.5B-Instruct, Apache-2.0) is
redistributed here; both are fetched from the Hugging Face Hub at run time.

## Repository structure

`main` holds the complete, self-contained snapshot archived at the DOI above:
source, tests, analysis scripts and all 24 run logs.

The weekly branches `week1`–`week6` are the laboratory record. Each holds the
state of the project at the end of that week, so the code that produced any
result is recoverable as it stood at the time. `week4-results` and `week5`
additionally hold the per-round model checkpoints, which are not part of the
archive.

# Dolly-15k FL × PEFT Research Placement

Federated fine-tuning of a small instruction-following language model under two
simultaneous constraints: limited client memory, which pushes towards quantising
the model, and non-identically distributed client data, which pushes clients
apart during training.

**Research question.** Does the performance cost of quantising the frozen base
model increase as client data become more non-identically distributed?

**Answer.** No effect of that kind was found. Across a full factorial grid of
24 runs, the cost of quantisation was effectively constant over a thousand-fold
change in data heterogeneity.

FOSE7901 STEM Research Placement, Macquarie University · Jeonghun Kim

---

## Key findings

- **Compression dominates; heterogeneity does not interact with it.** QLoRA
  4-bit cost about 5.7% in perplexity and QLoRA 8-bit about 0.5%, and those
  costs did not change as the data went from near-uniform to extremely uneven.
- **8-bit is lossless but slow.** 8-bit quantisation showed no measurable loss
  of quality, yet took more than twice as long per round as the uncompressed
  baseline. 4-bit ran at essentially baseline speed but cost quality.
- **Quantisation does not reduce communication.** Only the small LoRA adapter is
  transmitted, and it stays at full precision regardless of how the frozen base
  model is stored, so every run exchanged an identical 17.3 MB per round.

---

## Deliverables

Everything below is on the **`week6`** branch.

| Deliverable | File |
|---|---|
| Final report | `reports/Week6_Report_merged.docx` |
| Research records and data management | `reports/Week6_Research_Records.docx` |
| Presentation (3-minute format) | `reports/slides/Week6_3MT_slides.pptx`, `Week6_3MT_script.docx` |
| Figures used in the report | `reports/figures/report/` |
| Weekly progress reports | `reports/Week1_Report.docx` … `Week5_Report.docx` |

---

## How this repository is organised

There is no separate laboratory notebook. The repository is the record, in two
ways. Each week of the placement is kept as its own branch, so the state of the
code at any point in the project can be recovered exactly as it stood at the
time. And commit messages carry the reasoning behind each change, not only the
change itself, so the log reads as a decision record.

| Branch | What it holds |
|---|---|
| `week1` | Model and GPU environment verified; LoRA/QLoRA wiring |
| `week2` | Dolly-15k pipeline; Dirichlet partitioning |
| `week3` | FedProx and SCAFFOLD; checkpointing; convergence criterion; integration tests |
| `week4` | Runner and analysis functions for the core experimental grid |
| `week4-results` | **Data branch** — logs and checkpoints for the first 18 runs (413 MB) |
| `week5` | Design revision after supervisory feedback; server-side evaluation; the alpha = 100 runs and their logs (138 MB) |
| `week6` | Analysis, figures, final report, presentation, research records |

Each week branch carries its own README describing what was done that week.

---

## Weekly summary

| Week | What was done |
|---|---|
| 1 | Verified the model and GPU environment; wired up parameter-efficient fine-tuning (LoRA, QLoRA, DoRA) |
| 2 | Built the Dolly-15k data pipeline and Dirichlet partitioning across clients |
| 3 | Added FedProx and SCAFFOLD, checkpointing, the convergence criterion, and federated integration tests |
| 4 | Built the experiment runner and analysis functions; executed the first 18 runs on GPU |
| 5 | Revised the design after supervisory feedback; moved evaluation to the server; added and ran a fourth heterogeneity setting |
| 6 | Analysed all 24 runs; produced figures, the report, the research records and the presentation |

---

## Experimental design

A full factorial grid of 3 compression levels × 2 aggregation rules × 4 Dirichlet
settings, 24 runs in total, consuming 33.7 GPU-hours.

| Factor | Levels |
|---|---|
| Compression of the frozen base model | none (bfloat16), 8-bit, 4-bit NF4 |
| Aggregation rule | FedAvg, FedProx (mu = 0.01) |
| Data unevenness (Dirichlet alpha) | 0.1, 1, 10, 100 |

Model: Qwen2.5-0.5B-Instruct (Apache 2.0). Data: databricks-dolly-15k, 15,011
records (CC BY-SA 3.0). Eight clients, all participating every round, one local
epoch per round, up to ten rounds with early stopping.

---

## Reproducing the results

The analysis reads only the per-round JSONL logs, so every table and figure in
the report can be regenerated without a GPU and without repeating the training.

```bash
git clone https://github.com/hufs0529/peft_fl_STEM_placement.git
cd peft_fl_STEM_placement
git switch week6
git restore --source=origin/week4-results --worktree -- results/logs
git restore --source=origin/week5 --worktree -- results/logs
python scripts/analyze_interaction.py
python scripts/plot_results.py
```

The two `restore` commands bring the logs of all 24 runs into the working tree
without changing the branch. Analysis was run with Python 3.11.15, numpy 2.4.6,
rouge-score 0.1.2 and matplotlib 3.11.1; the training environment is recorded in
`env_alpha100.txt` on the `week5` branch.

To rerun the training itself, a GPU is required:

```bash
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 0.1
```

---

## Tests

```bash
pytest tests/
```

66 tests cover data preparation, partitioning, adapter wiring, aggregation,
checkpointing, the convergence criterion and the analysis functions. Two tests
are skipped outside a GPU environment because they exercise bitsandbytes
quantisation.