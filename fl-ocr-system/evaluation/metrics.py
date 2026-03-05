"""OCR evaluation metrics: Character Error Rate and Word Error Rate."""

from typing import Dict, List

import jiwer


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate using edit distance.

    CER = (S + D + I) / N where S=substitutions, D=deletions,
    I=insertions, N=total reference characters.
    """
    if not references:
        return 0.0

    # jiwer works at word level by default; for CER we split into characters
    pred_chars = [" ".join(list(p)) for p in predictions]
    ref_chars = [" ".join(list(r)) for r in references]

    try:
        cer = jiwer.wer(ref_chars, pred_chars)
    except ValueError:
        cer = 1.0

    return float(cer)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate."""
    if not references:
        return 0.0

    try:
        wer = jiwer.wer(references, predictions)
    except ValueError:
        wer = 1.0

    return float(wer)


def compute_all_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute CER and WER for a set of predictions."""
    return {
        "cer": compute_cer(predictions, references),
        "wer": compute_wer(predictions, references),
        "num_samples": len(predictions),
    }
