# 연합학습에서 양자화와 데이터 이질성은 서로를 악화시키는가

> 클라이언트 메모리 제약 때문에 모델을 양자화해야 하고, 동시에 기기마다 데이터 분포가 다른 상황.
> 이 두 악조건이 **곱해지는지(상호작용)** 아니면 **더해지기만 하는지**를 24개 실험으로 측정했습니다.

**결론: 더해지기만 합니다.** 양자화의 성능 대가는 데이터 이질성을 1000배 바꿔도 일정했습니다.
다만 이 null 결과는 **설계상 상호작용이 나타나기 어려운 조건이었다는 점**을 함께 보아야 합니다(→ [회고](#8-회고--무엇이-잘못되었나)).

| | |
|---|---|
| 기간 / 규모 | 6주 · 24개 학습 실행 · 33.7 GPU-시간 (NVIDIA A10G) |
| 모델 / 데이터 | Qwen2.5-0.5B-Instruct · databricks-dolly-15k (15,011건) |
| 스택 | PyTorch · Flower · PEFT · bitsandbytes · Transformers |
| 산출물 | 보고서, 연구기록 문서, 발표자료, 재현 가능한 로그·스크립트 |

---

## 1. 문제 정의

학습 데이터를 한곳에 모을 수 없는 상황이 늘고 있습니다. 병원 기록, 휴대폰의 입력 데이터처럼
기기 밖으로 내보낼 수 없는 데이터가 그렇습니다. **연합학습(Federated Learning)**은 데이터를
모으는 대신 모델을 각 기기로 보내 학습시키고, 결과만 합칩니다.

그런데 실제 클라이언트는 데이터센터가 아닙니다. 두 가지 제약이 동시에 걸립니다.

```
제약 1: 메모리가 부족하다        →  모델을 양자화해서 올린다 (QLoRA)
제약 2: 기기마다 데이터가 다르다  →  클라이언트 업데이트가 서로 어긋난다 (client drift)
```

각각의 해법은 잘 알려져 있습니다. 모르는 건 **둘이 겹칠 때 무슨 일이 일어나는가**입니다.

정밀도를 깎은 모델은 수치적 여유가 적으니, 서로 충돌하는 업데이트를 더 못 버틸 수 있습니다.
그렇다면 양자화의 대가는 데이터가 불균등할수록 커질 것입니다. **이것이 검증하려던 가설입니다.**

실무적으로 중요한 이유는 분명합니다. 상호작용이 있다면 압축 수준을 정할 때
사용자 데이터 분포를 알아야 하는데, 배포 시점에 그걸 아는 경우는 거의 없습니다.
없다면 압축은 **하드웨어만 보고** 정하면 됩니다.

---

## 2. 연구질문과 가설

> **RQ.** 동결 base 모델을 양자화할 때 생기는 성능 손실이, 클라이언트 데이터가
> 비균등해질수록 커지는가?

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

Everything below is on the **`main and week6`** branch.

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

From the Zenodo archive, unzip it and run:

    sha256sum -c MANIFEST.sha256
    python scripts/analyze_interaction.py
    python scripts/plot_results.py

From a clone of the repository, the same commands apply on `main`; no branch
switching is required.

    git clone https://github.com/hufs0529/peft_fl_STEM_placement.git
    cd peft_fl_STEM_placement

Analysis was run with Python 3.11.15, numpy 2.4.6, rouge-score 0.1.2 and
matplotlib 3.11.1. The training environment manifest is in `env_alpha100.txt`.

---

## Tests

```bash
pytest tests/
```

66 tests cover data preparation, partitioning, adapter wiring, aggregation,
checkpointing, the convergence criterion and the analysis functions. Two tests
are skipped outside a GPU environment because they exercise bitsandbytes
quantisation.
