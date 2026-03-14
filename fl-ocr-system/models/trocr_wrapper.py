"""TrOCR model wrapper for Sinhala handwritten OCR in a federated setting.

Aligns with the centralized training in notebooks/00_TrOCR_text_fine_tuned_handwritten.ipynb:
- Supports custom Sinhala model (danush99/Model_TrOCR-Sin-Printed-Text) with SinBERT tokenizer
  and DeiT image processor, or default TrOCR from Hugging Face.
- Same preprocessing, label handling (-100 for pad), and generation config (num_beams, length_penalty)
  as in the notebook for consistent evaluation.
"""

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


# Custom Sinhala TrOCR uses SinBERT + DeiT (same as notebook 00_TrOCR_text_fine_tuned_handwritten)
CUSTOM_SINHALA_MODEL_ID = "danush99/Model_TrOCR-Sin-Printed-Text"
CUSTOM_TOKENIZER_ID = "NLPC-UOM/SinBERT-large"
CUSTOM_IMAGE_PROCESSOR_ID = "facebook/deit-base-distilled-patch16-224"


class SinhalaOCRDataset(Dataset):
    """Dataset for Sinhala OCR image-text pairs.

    Matches notebook IAMDataset: RGB image, processor for pixel_values,
    tokenizer for labels with padding/max_length and -100 for pad tokens.
    """

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
        # Same as notebook: processor(image, return_tensors="pt") for single image
        out = self.processor(image, return_tensors="pt")
        pixel_values = out.pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so CrossEntropy ignores them (same as notebook)
        labels = labels.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def _build_custom_processor() -> TrOCRProcessor:
    """Build TrOCR processor from SinBERT tokenizer + DeiT image processor (notebook-aligned).
    Force image size to 224x224 so encoder position embeddings match (Hub preprocessor_config can use 256).
    """
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_ID)
    try:
        feature_extractor = ViTImageProcessor.from_pretrained(
            CUSTOM_IMAGE_PROCESSOR_ID, size=224
        )
    except TypeError:
        feature_extractor = ViTImageProcessor.from_pretrained(CUSTOM_IMAGE_PROCESSOR_ID)
        feature_extractor.size = 224
    return TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)


def _set_generation_config(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, max_length: int) -> None:
    """Set decoder tokens on model.config; generation params on model.generation_config.

    Newer Transformers rejects generation params (max_length, num_beams, etc.) on
    model.config; they must live in model.generation_config. Special tokens must
    also be in generation_config for encoder-decoder models.
    """
    decoder_start = processor.tokenizer.cls_token_id
    pad_id = processor.tokenizer.pad_token_id
    eos_id = getattr(processor.tokenizer, "sep_token_id", None) or processor.tokenizer.eos_token_id

    # Model config (for compatibility)
    model.config.decoder_start_token_id = decoder_start
    model.config.pad_token_id = pad_id
    model.config.vocab_size = model.config.decoder.vocab_size
    if eos_id is not None:
        model.config.eos_token_id = eos_id

    # Generation params: use generation_config (not model.config)
    model.generation_config = GenerationConfig(
        decoder_start_token_id=decoder_start,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )


class TrOCRWrapper:
    """Wrapper around HuggingFace TrOCR for loading, inference, and training.

    When model_name is the custom Sinhala printed model (danush99/Model_TrOCR-Sin-Printed-Text),
    loads the processor from SinBERT + DeiT to match the centralized training notebook.
    Otherwise uses the processor from the model's Hub repo (e.g. microsoft/trocr-base-handwritten).
    """

    def __init__(self, model_name: str = CUSTOM_SINHALA_MODEL_ID, max_length: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._use_custom_processor = model_name.strip().lower().endswith("model_trocr-sin-printed-text") or "danush99" in model_name

    def load(self, device: Optional[str] = None) -> "TrOCRWrapper":
        """Load the pretrained TrOCR model and processor (custom SinBERT+DeiT or from Hub)."""
        if self._use_custom_processor:
            self.processor = _build_custom_processor()
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        else:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        _set_generation_config(self.model, self.processor, self.max_length)

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
        """Run inference on a list of PIL images, returning decoded text.

        Uses model.config (num_beams=4, length_penalty=2.0, etc.) set at load time.
        """
        assert self.model is not None and self.processor is not None
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        self.model.eval()
        gc = self.model.generation_config
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=getattr(gc, "num_beams", 4),
                length_penalty=getattr(gc, "length_penalty", 2.0),
                early_stopping=getattr(gc, "early_stopping", True),
                no_repeat_ngram_size=getattr(gc, "no_repeat_ngram_size", 3),
            )

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
