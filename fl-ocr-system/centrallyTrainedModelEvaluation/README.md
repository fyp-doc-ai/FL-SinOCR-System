# Centralized Model Evaluation (Baseline Comparison)

This folder provides a script to evaluate the **centralized trained model** on the same handwritten test set used in FL evaluation, so you can compare FL global CER with the centralized baseline without training a new model.

## Centralized model

- **Hugging Face:** `danush99/Model_TrOCR-Sin-Handwritten-Text`
- Trained on the **same handwritten data** in a centralized way (see `notebooks/others/trocr-fine-tuned-handwritten.ipynb`).
- The model is **private**; you need a Hugging Face token with read access.

## Setup

1. Set your Hugging Face token (required for the private model):
   - **Environment:** `export HF_TOKEN=your_token`
   - **Or** create `fl-ocr-system/.env` with: `HF_TOKEN=your_token`
2. Ensure the test data path exists: `../allData/handwritten-data/test/` (relative to `fl-ocr-system/`) with `data.csv` and `images/`.

## Run evaluation

From the **fl-ocr-system** directory (with venv activated):

```bash
cd fl-ocr-system
source .venv/bin/activate
export HF_TOKEN=your_token   # if not using .env
python centrallyTrainedModelEvaluation/evaluate_centralized_model.py --test-dir ../allData/handwritten-data/test
```

Optional arguments:

- `--test-dir`: Path to the handwritten test directory (default: `../allData/handwritten-data/test`).
- `--batch-size`: Batch size for inference (default: 8).
- `--max-length`: Max decoder length (default: 64).

Output: CER and WER on the test set. Compare these with the final `global_cer` and `global_wer` from your FL run (last row of `experiments/results/.../metrics.csv`).
