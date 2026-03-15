# Research Manual: Federated Learning for Sinhala Handwritten OCR

This document is a step-by-step guide (A–Z) to run the system, perform evaluation, interpret results, and compare methods for your MSc thesis. It includes all commands and the execution flow needed to obtain meaningful research outcomes.

---

## Table of Contents

1. [Prerequisites and One-Time Setup](#1-prerequisites-and-one-time-setup)
2. [Execution Flow Overview](#2-execution-flow-overview)
3. [Step-by-Step: Data Partitioning](#3-step-by-step-data-partitioning)
4. [Step-by-Step: Federated Learning Experiments](#4-step-by-step-federated-learning-experiments)
5. [Step-by-Step: PEFT (Communication-Efficient) Experiments](#5-step-by-step-peft-experiments)
6. [Evaluation Process](#6-evaluation-process)
7. [Interpreting Results](#7-interpreting-results)
8. [Comparing Methods](#8-comparing-methods)
9. [Suggested Research Workflow for Thesis Impact](#9-suggested-research-workflow-for-thesis-impact)
10. [Command Reference](#10-command-reference)
11. [Troubleshooting and Tips](#11-troubleshooting-and-tips)
12. [Centralized Model Evaluation (Baseline Comparison)](#12-centralized-model-evaluation-baseline-comparison)

---

## 1. Prerequisites and One-Time Setup

### 1.1 Model and Centralized Baseline

The FL system uses the **same base model and preprocessing** as the centralized training notebook
`notebooks/others/trocr-fine-tuned-handwritten.ipynb`:

- **Model:** `danush99/Model_TrOCR-Sin-Printed-Text` (Sinhala printed TrOCR, fine-tuned for handwritten in that notebook).
- **Processor:** SinBERT tokenizer (`NLPC-UOM/SinBERT-large`) + DeiT image processor (`facebook/deit-base-distilled-patch16-224`) when this model is selected.
- **Evaluation:** Same generation config (num_beams=4, length_penalty=2.0) for CER/WER so results are comparable to the centralized run.

You can switch to `microsoft/trocr-base-handwritten` in `configs/base_config.yaml` (model.name) if needed; the processor is then loaded from the model’s Hub repo.

### 1.2 Environment

Ensure you have Python 3.8+ and the SinOCR dataset at `../allData/` (relative to `fl-ocr-system/`):

- `../allData/handwritten-data/train/` (images + `data.csv`)
- `../allData/handwritten-data/test/` (images + `data.csv`)
- `../allData/printed-data/train/` (images + `gt.csv`) — optional for institution partitioning
- `../allData/printed-data/test/` — optional

### 1.3 Virtual Environment and Dependencies

Run from the **project root** (`fl-ocr-system/`):

```bash
cd fl-ocr-system

# Create virtual environment
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
# .venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Keep the virtual environment activated for all following commands.

### 1.4 Verify Setup

```bash
# Check that key modules are importable
python -c "import flwr; import torch; import transformers; print('OK')"
```

---

## 2. Execution Flow Overview

High-level flow for meaningful results:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  A. PARTITION DATA (once per partition strategy)                         │
│     → data/partitions/client_0/, client_1/, ... + partition_summary.json  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  B. RUN FL EXPERIMENTS (one run per config)                             │
│     → experiments/results/{experiment_name}_{timestamp}/                 │
│       metrics.csv, summary.json, config.yaml, tb_logs/                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  C. EVALUATE & COMPARE                                                   │
│     → Notebooks 02 (partition viz), 03 (results analysis)               │
│     → Tables and plots for thesis                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Partitioning** defines how data is split across clients (Non-IID simulation).
- **Experiments** train the global model with a chosen FL algorithm and optional PEFT.
- **Evaluation** uses logged metrics (CER, WER, communication, fairness) and notebooks to interpret and compare.

---

## 3. Step-by-Step: Data Partitioning

Partitioning must be done **before** any FL experiment. The partition determines how many clients exist and how data is distributed (IID vs Non-IID).

### 3.1 Choose a Partition Method

| Method        | Use case                          | Config / script                    |
|---------------|-----------------------------------|------------------------------------|
| **Dirichlet** | Label-based Non-IID (controlled)  | `partition_by_dirichlet.py`        |
| **Clustering**| Writer-style Non-IID (visual)     | `partition_by_clustering.py`      |
| **Institution** | Mixed handwritten/printed      | `partition_by_institution.py`      |

### 3.2 Optional: Adjust Partition Settings

Edit `configs/base_config.yaml` if needed:

```yaml
partition:
  num_clients: 5           # number of FL clients
  alpha: 0.5               # Dirichlet: lower = more Non-IID (e.g. 0.1, 0.5, 1.0)
  num_clusters: 5          # for clustering method
  min_samples_per_client: 10
```

### 3.3 Run Partitioning

**Dirichlet (recommended for reproducibility and controlled Non-IID):**

```bash
python partition_scripts/partition_by_dirichlet.py --config configs/base_config.yaml
```

**Clustering (writer-level simulation):**

```bash
python partition_scripts/partition_by_clustering.py --config configs/base_config.yaml
```

**Institution-level (handwritten + printed mix):**

```bash
python partition_scripts/partition_by_institution.py --config configs/base_config.yaml
```

### 3.4 Verify Partition Output

- Check `data/partitions/partition_summary.json` for method, `num_clients`, and per-client sample counts.
- Check `data/partitions/client_0/`, `client_1/`, … for `data.csv`, `images/`, `metadata.json`.

**Visualize partitions (optional):** run `notebooks/02_partition_viz.ipynb` to see sample distribution and character distribution per client.

---

## 4. Step-by-Step: Federated Learning Experiments

Each experiment trains the global TrOCR model over multiple communication rounds using one FL algorithm. Results are written under `experiments/results/`.

### 4.1 Phase 1: FedAvg Baseline

**Purpose:** Establish baseline convergence and OCR performance.

```bash
python experiments/run_experiment.py --config configs/fedavg.yaml
```

- **What happens:** Server sends global model → clients train locally → server aggregates with FedAvg.
- **Output dir:** `experiments/results/fedavg_baseline_YYYYMMDD_HHMMSS/`.

### 4.2 Phase 2: SCAFFOLD (Non-IID Robustness)

**Purpose:** Compare against FedAvg under Non-IID data; expect better stability and fairness.

```bash
python experiments/run_experiment.py --config configs/scaffold.yaml
```

- **What happens:** Same as FedAvg plus control variates to reduce client drift.
- **Output dir:** `experiments/results/scaffold_noniid_YYYYMMDD_HHMMSS/`.

### 4.3 Phase 2: FedOPT (Server-Side Optimization)

**Purpose:** Compare server-side adaptive optimization (Adam) vs plain averaging.

```bash
python experiments/run_experiment.py --config configs/fedopt.yaml
```

- **What happens:** Server applies Adam to the aggregated update instead of simple averaging.
- **Output dir:** `experiments/results/fedopt_adam_YYYYMMDD_HHMMSS/`.

### 4.4 Important Notes for FL Runs

- **Same partition:** Use the same `data/partitions/` for all algorithm comparisons (run one partition script, then multiple experiment configs).
- **Reproducibility:** Seed is set from `configs/base_config.yaml` (`seed: 42`). Same config + same data → comparable runs.
- **Time:** Each run can take considerable time (e.g. 50 rounds × 3 clients × local epochs). Reduce `fl.num_rounds` or `training.local_epochs` in config for quick tests.

---

## 5. Step-by-Step: PEFT Experiments

PEFT reduces the number of trainable (and communicated) parameters. Use the **same partition** as for FL algorithm comparison.

### 5.1 LoRA (Low-Rank Adaptation)

```bash
python experiments/run_experiment.py --config configs/peft_lora.yaml
```

- **What happens:** Only LoRA parameters are trained and exchanged; full model is not sent.
- **Output dir:** `experiments/results/peft_lora_r8_YYYYMMDD_HHMMSS/`.

### 5.2 Adapter Modules

```bash
python experiments/run_experiment.py --config configs/peft_adapter.yaml
```

- **What happens:** Small adapter layers in the encoder are trained; rest is frozen.
- **Output dir:** `experiments/results/peft_adapter_dim64_YYYYMMDD_HHMMSS/`.

### 5.3 Encoder-Only Updates

```bash
python experiments/run_experiment.py --config configs/encoder_only.yaml
```

- **What happens:** Decoder is frozen; only encoder parameters are updated and communicated.
- **Output dir:** `experiments/results/encoder_only_YYYYMMDD_HHMMSS/`.

### 5.4 Comparing PEFT vs Full Fine-Tuning

For a fair comparison:

1. Run **FedAvg with full model** (`configs/fedavg.yaml`).
2. Run **same FL settings** with each PEFT method (LoRA, adapter, encoder_only).
3. Compare **CER/WER vs communication cost** (see Section 8).

---

## 6. Evaluation Process

Evaluation is **automatic** during each experiment: at every round (or every `eval_every_n_rounds`), the server evaluates the current global model and logs metrics.

### 6.1 Where Results Are Stored

Each run creates a timestamped directory under `experiments/results/`:

| File / folder   | Purpose |
|-----------------|--------|
| `metrics.csv`   | Per-round metrics (one row per round). |
| `summary.json`  | Final run summary (algorithm, PEFT, rounds, total communication, time). |
| `config.yaml`   | Full config used (for reproducibility). |
| `model_info.json` | Parameter counts (total, trainable). |
| `tb_logs/`      | TensorBoard logs (if enabled). |

### 6.2 Metrics in `metrics.csv`

| Column                | Meaning |
|-----------------------|--------|
| `round`               | Communication round index. |
| `global_cer`          | Character Error Rate on the **global test set** (handwritten test). |
| `global_wer`          | Word Error Rate on the global test set. |
| `mean_client_cer`     | Mean CER across clients (each evaluated on its own data). |
| `worst_client_cer`    | Maximum CER among clients (fairness metric). |
| `std_client_cer`     | Standard deviation of per-client CER. |
| `round_comm_mb`       | Communication (MB) in that round. |
| `cumulative_comm_mb`  | Total communication (MB) up to that round. |

### 6.3 OCR Metrics (CER and WER)

- **CER (Character Error Rate):** Edit distance (insertions, deletions, substitutions) at **character** level, normalized by total reference characters. Lower is better; 0 = perfect.
- **WER (Word Error Rate):** Same at **word** level. Typically higher than CER; lower is better.

Both are computed with the `jiwer` library on the model’s decoded text vs ground truth. Decoding uses the same generation config as the centralized notebook (num_beams=4, length_penalty=2.0) so FL evaluation is directly comparable to the baseline in `notebooks/others/trocr-fine-tuned-handwritten.ipynb`.

### 6.4 Running TensorBoard (Optional)

```bash
tensorboard --logdir experiments/results
```

Then open the URL shown (e.g. http://localhost:6006) to view scalar curves (e.g. global_cer, cumulative_comm_mb) across rounds and runs.

---

## 7. Interpreting Results

### 7.1 Convergence (Training Stability)

- **Plot:** `global_cer` (y) vs `round` (x).
- **Good:** CER decreases and stabilizes over rounds.
- **Concerning:** Strong oscillation or no improvement may indicate too few rounds, too few clients, or too aggressive learning rate / Non-IID severity.

### 7.2 Fairness (Non-IID Impact)

- **Plot:** `worst_client_cer` vs `round`.
- **Compare:** SCAFFOLD and FedOPT vs FedAvg. Lower worst-client CER and smaller gap between mean and worst client indicate better fairness under Non-IID.

### 7.3 Communication Efficiency (PEFT)

- **Plot:** `global_cer` (y) vs `cumulative_comm_mb` (x) for full model vs LoRA vs adapter vs encoder_only.
- **Interpretation:** A method that reaches similar CER with less cumulative MB is more communication-efficient. Trade-off: some PEFT methods may need more rounds to reach the same CER.

### 7.4 Summary Statistics

Use `summary.json` for each run:

- `algorithm`, `peft_method`: what was run.
- `total_rounds`: number of FL rounds.
- `total_mb` or `total_gb`: total communication.
- `elapsed_seconds`: wall-clock time.
- Final CER can be taken from the last row of `metrics.csv` (`global_cer`).

---

## 8. Comparing Methods

### 8.1 Algorithm Comparison (FedAvg vs SCAFFOLD vs FedOPT)

1. **Same partition and same number of rounds:** Use one partition (e.g. Dirichlet, alpha=0.5), then run `fedavg.yaml`, `scaffold.yaml`, `fedopt.yaml` without changing `data/partitions/`.
2. **Load all runs** in `notebooks/03_results_analysis.ipynb` (it discovers all directories under `experiments/results` that contain `metrics.csv`).
3. **Compare:**
   - Convergence: CER vs round for each algorithm.
   - Fairness: worst_client_cer vs round.
   - Final global CER and final worst_client_cer (table).

### 8.2 PEFT Comparison (Full vs LoRA vs Adapter vs Encoder-Only)

1. Run FedAvg + full model, then FedAvg + each PEFT config (same partition, same round count where possible).
2. **Compare:**
   - CER vs round (convergence).
   - CER vs cumulative_comm_mb (communication efficiency).
   - Final CER and total_mb from `summary.json` in a table.

### 8.3 Using the Results Notebook

In `notebooks/03_results_analysis.ipynb`:

1. Set `RESULTS_DIR = Path('../experiments/results')` (relative to notebook).
2. Run the cell that discovers experiments and loads `metrics.csv` and `summary.json`.
3. Use the provided plots:
   - Convergence: CER vs round.
   - Communication efficiency: CER vs cumulative MB.
   - Fairness: worst-client CER vs round.
4. Use the summary table cell to get a compact comparison (experiment name, algorithm, PEFT, rounds, final CER, total MB, time).

You can export figures and the table for your thesis.

### 8.4 Suggested Comparison Tables for Thesis

**Table 1 – FL algorithms (same partition, same rounds):**

| Algorithm | Final global CER | Final worst-client CER | Total communication (MB) | Time (s) |
|-----------|------------------|------------------------|---------------------------|----------|
| FedAvg    | 0.724            | 0.265                  | 275,561                   | 36,444   |
| SCAFFOLD  | …                | …                      | …                         | …        |
| FedOPT    | …                | …                      | …                         | …        |

**Table 2 – PEFT methods (same FL algorithm, same rounds):**

| PEFT method   | Final global CER | Total communication (MB) | Trainable params (%) |
|---------------|------------------|---------------------------|-----------------------|
| Full (none)   | 0.724            | 275,561                   | 100%                  |
| LoRA          | …                | …                         | &lt;1%                 |
| Adapter       | …                | …                         | small %               |
| Encoder-only  | …                | …                         | ~50%                  |

### 8.5 Comparing FL to centralized: use global CER

You do **not** need to train a new model in a centralized way to compare. A **centralized baseline** already exists: the model **`danush99/Model_TrOCR-Sin-Handwritten-Text`** was trained with the same handwritten data in a centralized manner (via the notebook `notebooks/others/trocr-fine-tuned-handwritten.ipynb`). You can run the **same evaluation** (same test set, same CER/WER setup) on that model and compare its CER to the **final global CER** from your FL run. The gap (FL global CER − centralized CER) tells you how close federated learning is to the centralized baseline. See [Section 12](#12-centralized-model-evaluation-baseline-comparison) for how to run this evaluation.

---

## 9. Suggested Research Workflow for Thesis Impact

1. **Setup and partition (once)**  
   - Create venv, install dependencies, verify data.  
   - Run **Dirichlet** partitioning (e.g. alpha=0.5, 5 clients).  
   - Optionally run clustering or institution partitioning for ablations.

2. **Phase 1 – Federated system**  
   - Run **FedAvg** baseline; confirm convergence and reasonable CER.  
   - Note final CER, worst-client CER, and total communication.

3. **Phase 2 – Non-IID robustness**  
   - Run **SCAFFOLD** and **FedOPT** with the **same** partition and round count.  
   - Compare convergence, final CER, worst-client CER, and stability.  
   - Use notebook 03 to plot and fill Table 1.

4. **Phase 3 – Communication efficiency**  
   - Run **LoRA**, **adapter**, and **encoder_only** with the same FL algorithm (e.g. FedAvg) and same partition.  
   - Compare CER vs round and CER vs cumulative MB; fill Table 2.

5. **Ablations (optional)**  
   - Different `partition.alpha` (e.g. 0.1, 0.5, 1.0) with FedAvg and SCAFFOLD.  
   - Different `fl.num_rounds` or `training.local_epochs` to discuss convergence vs cost.

6. **Write-up**  
   - Use `config.yaml` and `summary.json` for reproducibility.  
   - Use metrics.csv and the notebooks for figures and tables.  
   - Report mean ± std if you run multiple seeds.

---

## 10. Command Reference

All commands assume the virtual environment is activated and the current directory is `fl-ocr-system/`.

### Environment

```bash
cd fl-ocr-system
python3 -m venv .venv
source .venv/bin/activate                    # macOS/Linux
# .venv\Scripts\Activate.ps1                 # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Partitioning

```bash
# Dirichlet (label-based Non-IID)
python partition_scripts/partition_by_dirichlet.py --config configs/base_config.yaml

# Clustering (writer-style)
python partition_scripts/partition_by_clustering.py --config configs/base_config.yaml

# Institution (handwritten + printed mix)
python partition_scripts/partition_by_institution.py --config configs/base_config.yaml
```

### FL Experiments

```bash
# Phase 1: Baseline
python experiments/run_experiment.py --config configs/fedavg.yaml

# Phase 2: Non-IID robustness
python experiments/run_experiment.py --config configs/scaffold.yaml
python experiments/run_experiment.py --config configs/fedopt.yaml

# Phase 3: PEFT
python experiments/run_experiment.py --config configs/peft_lora.yaml
python experiments/run_experiment.py --config configs/peft_adapter.yaml
python experiments/run_experiment.py --config configs/encoder_only.yaml
```

### Evaluation and Visualization

```bash
# TensorBoard (from fl-ocr-system)
tensorboard --logdir experiments/results
```

Then run Jupyter for notebooks:

```bash
jupyter notebook notebooks/
# Open 01_data_analysis.ipynb, 02_partition_viz.ipynb, 03_results_analysis.ipynb
```

### Optional: Hyperparameter Sweep

```bash
# Generate sweep configs only (dry run)
python experiments/sweep.py --base-config configs/base_config.yaml --output-dir configs/sweep --dry-run

# Run full sweep (many experiments)
python experiments/sweep.py --base-config configs/base_config.yaml --output-dir configs/sweep
```

---

## 11. Troubleshooting and Tips

- **“No client partitions found”**  
  Run one of the partition scripts first and ensure `data/partitions/client_0/`, etc., exist.

- **Out-of-memory**  
  Reduce `training.batch_size` in config (e.g. to 4). Reduce `fl.clients_per_round` so fewer clients are active per round.

- **Slow runs**  
  Reduce `fl.num_rounds` (e.g. 10) or `training.local_epochs` (e.g. 1) for quick tests. Use GPU if available (automatic when `torch.cuda.is_available()`).

- **Reproducibility**  
  Keep `seed` in `base_config.yaml` fixed; do not change partition or config between runs you want to compare.

- **Data path**  
  Config uses `data.base_dir: "../allData"`. If your dataset is elsewhere, override in the YAML or set paths in `base_config.yaml`.

- **Comparing runs**  
  Use the **same** `data/partitions/` for all runs you compare. Only change the experiment config (algorithm, PEFT, or hyperparameters).

---

## 12. Centralized Model Evaluation (Baseline Comparison)

You do **not** need to train a new model in a centralized way. A **centralized trained model** already exists and was trained on the same handwritten data using the notebook `notebooks/others/trocr-fine-tuned-handwritten.ipynb`:

- **Model on Hugging Face:** `danush99/Model_TrOCR-Sin-Handwritten-Text`

By evaluating this model on the **same test set** (handwritten test) with the **same** CER/WER setup (num_beams=4, length_penalty=2.0), you get a **centralized baseline CER**. Compare that to the **final global CER** from your FL run to see how close federated learning is to centralized training.

### 12.1 Accessing the model (private)

The model is **private** on Hugging Face. You need to authenticate with a **Hugging Face token** that has read access to this repo:

1. Set your token in the environment or in a `.env` file under `fl-ocr-system/` (do not commit the token):
   - **Environment:** `export HF_TOKEN=your_token`
   - **Or** create `fl-ocr-system/.env` with a line: `HF_TOKEN=your_token`
2. Obtain the token from [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) if needed.

### 12.2 Running the evaluation

The folder **`centrallyTrainedModelEvaluation/`** contains a script to load the centralized model and compute CER/WER on the handwritten test set (same evaluation as in FL):

```bash
cd fl-ocr-system
source .venv/bin/activate
export HF_TOKEN=your_token   # or use .env
python centrallyTrainedModelEvaluation/evaluate_centralized_model.py --test-dir ../allData/handwritten-data/test
```

Output: CER and WER on the test set, so you can compare directly with the final `global_cer` / `global_wer` from `experiments/results/.../metrics.csv`.

---

This manual, together with the README and the plan, should be enough to run the system end-to-end, evaluate results, and compare methods systematically for your thesis.
