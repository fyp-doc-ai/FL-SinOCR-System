"""Evaluate the centralized trained TrOCR model on the handwritten test set.

This gives a baseline CER/WER comparable to FL global_cer/global_wer (same test set,
same generation config). Run from fl-ocr-system with HF_TOKEN set for the private model.

Usage:
    cd fl-ocr-system && python centrallyTrainedModelEvaluation/evaluate_centralized_model.py --test-dir ../allData/handwritten-data/test
"""

import argparse
import os
import sys
from pathlib import Path

# Add fl-ocr-system root so we can import evaluation and partition_scripts
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _load_env() -> None:
    """Load HF_TOKEN from .env if present."""
    env_file = _PKG_ROOT / ".env"
    if env_file.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        except ImportError:
            pass
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip().strip('"').strip("'")
                        if key == "HF_TOKEN" or key == "HUGGING_FACE_HUB_TOKEN":
                            os.environ[key] = value
                            break


def main() -> None:
    _load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set. Set it in the environment or in fl-ocr-system/.env")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Evaluate centralized TrOCR on handwritten test set")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(_PKG_ROOT.parent / "allData" / "handwritten-data" / "test"),
        help="Path to handwritten test directory (data.csv + images/)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-length", type=int, default=64, help="Max decoder length")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        GenerationConfig,
        TrOCRProcessor,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
    )

    from evaluation.metrics import compute_all_metrics
    from models.trocr_wrapper import SinhalaOCRDataset
    from partition_scripts.partition_utils import load_dataset_csv

    model_id = "danush99/Model_TrOCR-Sin-Handwritten-Text"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token_kw = {"token": token}

    # Build processor the same way as the working Flask app: SinBERT + google/vit-base-patch16-224
    # (Do not use TrOCRProcessor.from_pretrained(model_id) — the handwritten model was trained with this setup.)
    print("Loading tokenizer and image processor (SinBERT + ViT)...")
    tokenizer = AutoTokenizer.from_pretrained("NLPC-UOM/SinBERT-large", **token_kw)
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", **token_kw)
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    print(f"Loading model {model_id} (private; using HF_TOKEN)...")
    model = VisionEncoderDecoderModel.from_pretrained(model_id, **token_kw)

    # Set decoder/generation config like the Flask app (required for correct generation)
    cls_id = processor.tokenizer.cls_token_id
    sep_id = processor.tokenizer.sep_token_id
    pad_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = cls_id
    model.config.pad_token_id = pad_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = sep_id
    model.generation_config = GenerationConfig(
        max_length=args.max_length,
        early_stopping=True,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        decoder_start_token_id=cls_id,
        eos_token_id=sep_id,
        pad_token_id=pad_id,
    )

    model.to(device)
    model.eval()

    print(f"Loading test data from {args.test_dir}...")
    image_paths, texts = load_dataset_csv(args.test_dir)
    print(f"  Samples: {len(image_paths)}")

    dataset = SinhalaOCRDataset(image_paths, texts, processor, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values,
                decoder_start_token_id=processor.tokenizer.cls_token_id,
            )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(decoded)

    metrics = compute_all_metrics(all_predictions, texts)
    print("\n--- Centralized model (same test set as FL) ---")
    print(f"  CER: {metrics['cer']:.6f}")
    print(f"  WER: {metrics['wer']:.6f}")
    print(f"  Num samples: {metrics['num_samples']}")
    print("\nCompare the CER above with the final global_cer in experiments/results/.../metrics.csv (last row).")


if __name__ == "__main__":
    main()
