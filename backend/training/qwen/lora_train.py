#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TextDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], tokenizer, max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        row = self.items[index]
        architecture = json.dumps(row["architecture"], ensure_ascii=True)
        target = row["explanation"]

        prompt = (
            "You are an AI architecture assistant. Explain briefly and clearly.\n"
            f"Architecture JSON: {architecture}\n"
            "Explanation:"
        )

        text = prompt + " " + target
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training for Qwen2.5-1.5B.")
    parser.add_argument("--dataset", default="data/synthetic/dataset.jsonl")
    parser.add_argument("--output", default="checkpoints/qwen_lora")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-train-samples", type=int, default=256)
    return parser.parse_args()


def load_dataset(path: Path, limit: int) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "architecture" not in row or "explanation" not in row:
                continue
            items.append({"architecture": row["architecture"], "explanation": row["explanation"]})
            if len(items) >= limit:
                break
    return items


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this training script.")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_dataset(dataset_path, args.max_train_samples)
    if not rows:
        raise RuntimeError("Dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    model.train()

    ds = TextDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=max(1, min(args.batch_size, 2)), shuffle=True)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += float(loss.item()) * args.grad_accum

            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        print(f"epoch={epoch + 1} avg_loss={running / max(1, len(loader)):.4f} steps={step}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
