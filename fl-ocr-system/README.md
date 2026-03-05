# Federated Learning for Sinhala Handwritten OCR

MSc research implementation: applying Federated Learning to improve Sinhala handwritten OCR
using a pretrained TrOCR (Vision Transformer Encoder + Transformer Decoder) architecture.

## Research Objectives

1. **Federated System Design** — baseline FL training with FedAvg
2. **Robust Federated Training under Non-IID Data** — SCAFFOLD, FedOPT
3. **Communication-Efficient Fine-Tuning** — LoRA, adapters, encoder-only updates

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Partition the dataset (Dirichlet-based Non-IID)
python partition_scripts/partition_by_dirichlet.py --config configs/base_config.yaml

# Run a FedAvg experiment
python experiments/run_experiment.py --config configs/fedavg.yaml

# Run SCAFFOLD
python experiments/run_experiment.py --config configs/scaffold.yaml

# Run with LoRA
python experiments/run_experiment.py --config configs/peft_lora.yaml
```

---

## Project Structure: Folders and Files

### Root

| File | Purpose |
|------|---------|
| **README.md** | This file — project overview, usage, and structure. |
| **requirements.txt** | Python dependencies (flwr, torch, transformers, peft, jiwer, etc.). |

---

### `configs/`

**Purpose:** YAML configuration files for experiments. All experiments inherit from `base_config.yaml` and override algorithm- or PEFT-specific settings.

| File | Purpose |
|------|---------|
| **base_config.yaml** | Shared defaults: data paths, partition settings, model name, FL params (rounds, clients per round), training (epochs, LR, batch size), PEFT options, server optimizer, logging. |
| **fedavg.yaml** | FedAvg baseline experiment: `algorithm: fedavg`, `peft.method: none`. |
| **scaffold.yaml** | SCAFFOLD experiment: `algorithm: scaffold`, control-variate server LR. |
| **fedopt.yaml** | FedOPT experiment: `algorithm: fedopt`, server-side Adam/Adagrad settings. |
| **peft_lora.yaml** | LoRA experiment: `peft.method: lora`, rank/alpha/dropout and target modules. |
| **peft_adapter.yaml** | Adapter experiment: `peft.method: adapter`, bottleneck dim and dropout. |
| **encoder_only.yaml** | Encoder-only experiment: `peft.method: encoder_only` (decoder frozen). |

---

### `data/`

**Purpose:** Generated federated client data. Populated by partition scripts; not committed with source (only structure).

| Path | Purpose |
|------|---------|
| **data/partitions/** | Output directory for partition scripts. Contains one subfolder per client (`client_0/`, `client_1/`, …) plus `partition_summary.json`. |

Each `client_{id}/` contains:
- **images/** — copied image files for that client
- **data.csv** — `file_name, text` for local training
- **metadata.json** — `num_samples`, `partition_type`, and method-specific fields (e.g. `alpha`, `cluster_id`)

---

### `partition_scripts/`

**Purpose:** Split the SinOCR dataset into client-specific subsets to simulate Non-IID federated environments (writer-level, label-based, or institution-level).

| File | Purpose |
|------|---------|
| **partition_utils.py** | Shared helpers: load dataset from `data.csv`/`gt.csv`, save client folders (images + CSV + metadata), build character-label map for Dirichlet, partition summary JSON. |
| **partition_by_dirichlet.py** | Dirichlet-based partitioning: assign samples to clients by label (first-character) distribution; low `alpha` = more Non-IID. CLI: `--config configs/base_config.yaml`. |
| **partition_by_clustering.py** | Writer-level simulation: extract simple visual features, K-Means cluster, one cluster per client. CLI: `--config configs/base_config.yaml`. |
| **partition_by_institution.py** | Institution-level: mix handwritten-only, printed-only, and mixed clients with uneven sizes. CLI: `--config configs/base_config.yaml`. |
| **__init__.py** | Package marker. |

---

### `models/`

**Purpose:** TrOCR model loading, dataset definition, and parameter utilities for FL (get/set trainable params, byte count, freeze/unfreeze).

| File | Purpose |
|------|---------|
| **trocr_wrapper.py** | `TrOCRWrapper`: load `microsoft/trocr-base-handwritten` and processor, create datasets, run inference (`generate`). `SinhalaOCRDataset`: image–text pairs with processor and max length. |
| **model_utils.py** | `get_parameters_as_ndarrays` / `set_parameters_from_ndarrays` (trainable-only option for PEFT), `compute_parameter_bytes`, `freeze_module` / `unfreeze_module`, `count_parameters`. |
| **__init__.py** | Package marker. |

---

### `peft_modules/`

**Purpose:** Parameter-Efficient Fine-Tuning — LoRA, adapters, and encoder-only updates to reduce communication and training cost.

| File | Purpose |
|------|---------|
| **peft_utils.py** | `apply_peft(model, cfg)`: dispatches to the configured method (`none`, `lora`, `adapter`, `encoder_only`) and prints parameter stats. |
| **lora.py** | `apply_lora`: inject LoRA into TrOCR via HuggingFace PEFT (`LoraConfig`, `get_peft_model`) on encoder attention (e.g. query, value). |
| **adapters.py** | `AdapterLayer` (bottleneck down–up with residual), `AdapterWrappedLayer`; `apply_adapters`: insert adapters after encoder blocks and freeze the rest. |
| **encoder_only.py** | `apply_encoder_only`: freeze full model then unfreeze encoder only; ~50% fewer trainable parameters. |
| **__init__.py** | Package marker. |

---

### `fl_server/`

**Purpose:** Federated learning server — strategy, client selection, and pluggable aggregation (FedAvg, SCAFFOLD, FedOPT).

| File | Purpose |
|------|---------|
| **server.py** | `create_aggregator(cfg, num_model_params)`: build FedAvg/SCAFFOLD/FedOPT from config. `FLStrategy`: Flower `Strategy` that uses the aggregator for `aggregate_fit`, and handles `configure_fit`, `configure_evaluate`, `aggregate_evaluate`, optional server `evaluate_fn`. |
| **aggregators/__init__.py** | Exports `AggregatorInterface`, `FedAvgAggregator`, `ScaffoldAggregator`, `FedOptAggregator`, and `AGGREGATOR_REGISTRY`. |
| **aggregators/base.py** | `AggregatorInterface` ABC (`aggregate_fit`, `get_name`) and `weighted_average(results)` for parameter aggregation. |
| **aggregators/fedavg.py** | `FedAvgAggregator`: weighted average of client parameters by `num_examples`. |
| **aggregators/scaffold.py** | `ScaffoldAggregator`: maintains global control variate; aggregates model params and control-variate updates; `get_global_control()` for client config. |
| **aggregators/fedopt.py** | `FedOptAggregator`: server-side optimizer (Adam/Adagrad/SGD) on aggregated pseudo-gradient; keeps momentum/variance state. |
| **__init__.py** | Package marker. |

---

### `fl_clients/`

**Purpose:** Flower clients that train TrOCR on local partitions and exchange only (trainable) parameters with the server.

| File | Purpose |
|------|---------|
| **client.py** | `TrOCRFlowerClient(fl.client.NumPyClient)`: `get_parameters` / `set_parameters` (trainable-only for PEFT), `fit` (local training, optional SCAFFOLD delta_c), `evaluate`. `create_client_fn(model_factory, processor, cfg, device)`: returns a function that, given client id, loads partition data and builds a client instance. |
| **client_utils.py** | `load_client_data(partition_dir, client_id)`: image paths and texts from `client_{id}/data.csv` and `images/`. `load_client_metadata`, `create_client_dataloader`, `get_num_clients`. |
| **__init__.py** | Package marker. |

---

### `training/`

**Purpose:** Local (on-client) training loop and learning-rate scheduling.

| File | Purpose |
|------|---------|
| **trainer.py** | `LocalTrainer`: optimizer (AdamW/SGD), gradient clipping; `train(dataloader, epochs)` returns average loss; `evaluate(dataloader)` returns loss and metrics dict. |
| **lr_scheduler.py** | `get_cosine_schedule`, `get_linear_schedule`, `get_constant_schedule` for optimizer LR. |
| **__init__.py** | Package marker. |

---

### `evaluation/`

**Purpose:** OCR metrics (CER/WER), global and per-client evaluation, and communication cost tracking.

| File | Purpose |
|------|---------|
| **metrics.py** | `compute_cer`, `compute_wer` (using `jiwer`), `compute_all_metrics` for prediction/reference lists. |
| **eval_pipeline.py** | `evaluate_global_model`: run model on full test set, return CER/WER. `evaluate_per_client`: CER per client, aggregated stats (mean, std, min, max, worst-client CER). |
| **communication_cost.py** | `CommunicationTracker`: uses trainable parameter byte count; `log_round(server_round, num_clients_fit, num_clients_eval)` for per-round and cumulative bytes/MB; `get_summary`, `get_all_round_logs`. |
| **__init__.py** | Package marker. |

---

### `experiments/`

**Purpose:** Entry points to run single experiments or hyperparameter sweeps.

| File | Purpose |
|------|---------|
| **run_experiment.py** | Main entry: load config, set seed, build model + PEFT, create aggregator and FL strategy, set up server-side evaluation (global + per-client CER, communication logging), run `flwr.simulation.start_simulation` with `client_fn`. Logs config, model info, round metrics, final summary. CLI: `--config configs/fedavg.yaml` (or any experiment YAML). |
| **sweep.py** | Grid over `fl.algorithm`, `peft.method`, `partition.alpha`; generates YAMLs under `configs/sweep/` and optionally runs each. CLI: `--base-config`, `--output-dir`, `--dry-run`. |
| **__init__.py** | Package marker. |

**experiments/results/** — Created at runtime; holds one timestamped directory per run with `metrics.csv`, `config.yaml`, `summary.json`, `model_info.json`, and `tb_logs/` when TensorBoard is enabled.

---

### `logging_utils/`

**Purpose:** Centralized experiment logging (CSV, TensorBoard, optional W&B).

| File | Purpose |
|------|---------|
| **logger.py** | `ExperimentLogger(cfg)`: writes round metrics to CSV, TensorBoard (if enabled), and optionally Weights & Biases. Methods: `log_round(metrics)`, `log_config`, `log_model_info`, `log_final_summary`, `close`. Output directory: `logging.output_dir / {experiment_name}_{timestamp}`. |
| **__init__.py** | Package marker. |

---

### `notebooks/`

**Purpose:** Exploratory analysis and result visualization (not used for training or evaluation scripts).

| File | Purpose |
|------|---------|
| **01_data_analysis.ipynb** | SinOCR exploration: load train/test CSV for handwritten and printed data, text length distributions, character frequency (top characters), sample image grid. |
| **02_partition_viz.ipynb** | After partitioning: load `partition_summary.json`, bar chart of samples per client, character-distribution heatmap per client, text-length distribution per client. |
| **03_results_analysis.ipynb** | Load experiment results from `experiments/results/`: discovery of runs, convergence (CER vs round), communication efficiency (CER vs cumulative MB), worst-client CER, summary table (algorithm, PEFT, rounds, final CER, total MB, time). |

---

## Dataset

The SinOCR dataset is expected at `../allData/` relative to this directory:

- **handwritten-data/train/** — 908 training images + `data.csv` (`file_name`, `text`)
- **handwritten-data/test/** — 227 test images + `data.csv`
- **printed-data/train/** — 90,000 training images + `gt.csv`
- **printed-data/test/** — 10,000 test images + `gt.csv`

---

## Switching Algorithms

Set `fl.algorithm` in the config (or in experiment YAMLs that override base):

- **fedavg** — Federated Averaging (baseline)
- **scaffold** — SCAFFOLD with control variates
- **fedopt** — Server-side adaptive optimization (Adam/Adagrad/SGD)

---

## PEFT Strategies

Set `peft.method` in config:

- **none** — full model fine-tuning
- **lora** — Low-Rank Adaptation (encoder attention)
- **adapter** — bottleneck adapter layers in encoder
- **encoder_only** — freeze decoder, train encoder only
