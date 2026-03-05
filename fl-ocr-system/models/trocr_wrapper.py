"""TrOCR model wrapper for Sinhala handwritten OCR in a federated setting."""

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


class SinhalaOCRDataset(Dataset):
    """Dataset for Sinhala OCR image-text pairs."""

    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        processor: TrOCRProcessor,
        max_length: int = 64,
    ):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so CrossEntropy ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


class TrOCRWrapper:
    """Wrapper around HuggingFace TrOCR for loading, inference, and training."""

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", max_length: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None

    def load(self, device: Optional[str] = None) -> "TrOCRWrapper":
        """Load the pretrained TrOCR model and processor."""
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.max_length = self.max_length

        if device:
            self.model.to(device)

        return self

    def create_dataset(
        self,
        image_paths: List[str],
        texts: List[str],
    ) -> SinhalaOCRDataset:
        """Create a dataset from image paths and text labels."""
        assert self.processor is not None, "Call load() first"
        return SinhalaOCRDataset(image_paths, texts, self.processor, self.max_length)

    def generate(self, images: List[Image.Image], device: str = "cpu") -> List[str]:
        """Run inference on a list of PIL images, returning decoded text."""
        assert self.model is not None and self.processor is not None
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=self.max_length)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def get_trainable_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """Return list of (name, param) that require gradients."""
        assert self.model is not None
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def get_trainable_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for _, p in self.get_trainable_parameters())

    def get_total_param_count(self) -> int:
        """Count all parameters."""
        assert self.model is not None
        return sum(p.numel() for p in self.model.parameters())
