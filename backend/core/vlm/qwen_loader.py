"""
Qwen2.5-3B-Instruct loader with 4-bit quantisation.

Uses ``transformers`` + ``bitsandbytes`` for memory-efficient inference on
consumer hardware.  The model and tokeniser are cached after the first call.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)

# Model identifier on HuggingFace Hub.
QWEN_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"


@lru_cache(maxsize=1)
def load_qwen() -> Tuple[object, object]:
    """Load Qwen2.5-3B-Instruct with 4-bit quantisation.

    The model is loaded once per process and returned from cache on subsequent
    calls.

    Returns:
        ``(model, tokenizer)`` tuple.

    Raises:
        ImportError: If ``transformers`` or ``bitsandbytes`` are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers and bitsandbytes are required. "
            "Install with: pip install transformers bitsandbytes accelerate"
        ) from exc

    logger.info("Loading %s with 4-bit quantisation…", QWEN_MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Qwen2.5-3B-Instruct loaded successfully.")
    return model, tokenizer
