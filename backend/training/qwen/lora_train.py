"""
LoRA fine-tuning of Qwen2.5-3B-Instruct for architecture explanation.

Uses HuggingFace ``peft`` (LoRA) + ``transformers`` Trainer on paired
(diagram image embedding + architecture JSON, explanation text) data from the
synthetic JSONL manifest.

The training task is next-token prediction (causal LM) on the full formatted
chat turn:
    [SYSTEM prompt] [USER: arch description] [ASSISTANT: explanation]

Vision embeddings are stored in the JSONL but are not yet fused into the
prompt at this stage — that multi-modal connection is handled by the
VisionProjector in the next training phase.

Usage::

    python -m backend.training.qwen.lora_train \
        --data  data/synthetic \
        --out   checkpoints/qwen_lora \
        --epochs 3 \
        --lr 2e-4
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ExplanationDataset(Dataset):
    """Tokenised dataset for Qwen LoRA fine-tuning.

    Each sample is a fully formatted chat turn, tokenised and truncated to
    *max_length* tokens.  The loss is computed only on the assistant reply
    (labels for the prompt tokens are set to -100).
    """

    def __init__(
        self,
        manifest_path: Path,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []

        with manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line.strip())
                if entry.get("explanation"):
                    self.samples.append(entry)

        logger.info("ExplanationDataset: %d samples", len(self.samples))

    def _format(self, entry: Dict) -> str:
        from backend.core.vlm.explainer import _build_messages  # lazy import
        from backend.api.schemas.architecture import Architecture

        arch = Architecture.model_validate(entry["architecture"])
        messages = _build_messages(arch)
        # Append the ground-truth assistant response.
        messages.append({"role": "assistant", "content": entry["explanation"]})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self._format(self.samples[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: clone input_ids, mask padding.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    """Fine-tune Qwen2.5-3B-Instruct with LoRA on explanation data.

    Args:
        data_dir:      Root directory containing ``dataset.jsonl``.
        out_dir:       Output directory for LoRA adapter weights.
        epochs:        Number of full passes over the dataset.
        lr:            AdamW learning rate.
        batch_size:    Per-device batch size.
        grad_accum:    Gradient accumulation steps.
        max_length:    Maximum sequence length in tokens.
        lora_r:        LoRA rank.
        lora_alpha:    LoRA scaling factor.
        lora_dropout:  LoRA dropout rate.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        from transformers import TrainingArguments, Trainer      # type: ignore
    except ImportError as exc:
        raise ImportError(
            "peft and transformers are required. "
            "Install with: pip install peft transformers accelerate"
        ) from exc

    from backend.core.vlm.qwen_loader import load_qwen

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "dataset.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"JSONL manifest not found: {manifest}. "
            "Run scripts/generate_dataset.py first."
        )

    # ── Load model & tokeniser ───────────────────────────────────────────────
    model, tokenizer = load_qwen()

    # ── LoRA config ──────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Required for gradient checkpointing to work with LoRA adapters.
    model.enable_input_require_grads()

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = ExplanationDataset(manifest, tokenizer, max_length=max_length)

    val_size  = max(1, int(len(ds) * 0.1))
    train_size = len(ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    # ── TrainingArguments ────────────────────────────────────────────────────
    args_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "logging_steps": 10,
        "report_to": "none",
        "dataloader_num_workers": 2,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**args_kwargs)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logger.info("Starting LoRA fine-tuning of Qwen2.5-3B-Instruct…")
    trainer.train()

    # Save LoRA adapter only (small checkpoint).
    adapter_path = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved → %s", adapter_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Qwen for ArchitectAI.")
    p.add_argument("--data",    default="data/synthetic",       help="Data root dir.")
    p.add_argument("--out",     default="checkpoints/qwen_lora", help="Output dir.")
    p.add_argument("--epochs",  type=int,   default=3,    help="Training epochs.")
    p.add_argument("--lr",      type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--bs",      type=int,   default=4,    help="Per-device batch size.")
    p.add_argument("--accum",   type=int,   default=4,    help="Gradient accumulation steps.")
    p.add_argument("--maxlen",  type=int,   default=512,  help="Max sequence length.")
    p.add_argument("--lora_r",  type=int,   default=16,   help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=32,  help="LoRA alpha.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=Path(args.data),
        out_dir=Path(args.out),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.bs,
        grad_accum=args.accum,
        max_length=args.maxlen,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
