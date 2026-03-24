"""
Qwen2.5-3B-Instruct loader with 4-bit quantisation.

Uses ``transformers`` + ``bitsandbytes`` for memory-efficient inference on
consumer hardware.  The model and tokeniser are cached after the first call.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)

# Preferred model identifier on HuggingFace Hub.
QWEN_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
QWEN_FALLBACK_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Keep VRAM usage under control on 6GB GPUs by allowing CPU/disk offload.
    # This avoids OOM spikes during checkpoint materialization.
    offload_dir = Path(".hf_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)

    requested_model = os.getenv("ARCHITECTAI_QWEN_MODEL", QWEN_MODEL_ID).strip()
    candidates = [requested_model]
    if requested_model != QWEN_FALLBACK_MODEL_ID:
        candidates.append(QWEN_FALLBACK_MODEL_ID)

    for model_id in candidates:
        logger.info("Loading %s with 4-bit quantisation…", model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "4GiB", "cpu": "24GiB"},
                offload_folder=str(offload_dir),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.eval()
            logger.info("%s loaded successfully.", model_id)
            return model, tokenizer
        except torch.OutOfMemoryError:
            logger.warning(
                "CUDA OOM while loading %s; trying a smaller fallback model.",
                model_id,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        "Failed to load a Qwen model on available hardware. "
        "Set ARCHITECTAI_QWEN_MODEL to a smaller checkpoint."
    )
