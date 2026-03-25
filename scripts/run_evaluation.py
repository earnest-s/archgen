#!/usr/bin/env python3
import difflib
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_dataset(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "architecture" in row and "explanation" in row:
                rows.append(row)
    return rows


def build_prompt(architecture: Dict[str, object]) -> str:
    return (
        "You are an AI architecture assistant. Explain this architecture briefly and clearly.\n"
        f"Architecture JSON: {json.dumps(architecture, ensure_ascii=True)}\n"
        "Explanation:"
    )


def compute_bleu(reference: str, hypothesis: str) -> float:
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA model with BLEU-4 and ROUGE-L.")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")
    print("[STEP 3/3] Evaluation started")

    dataset_path = Path("data/synthetic/dataset.jsonl")
    adapter_path = Path("checkpoints/qwen_lora")
    report_path = Path("reports/evaluation.json")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for evaluation.")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_path}")

    rows = load_dataset(dataset_path)
    rows = rows[: max(1, args.max_samples)]
    if not rows:
        raise RuntimeError("Dataset is empty.")
    print(f"[INFO] Loaded {len(rows)} evaluation samples from {dataset_path}")

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_scores: List[float] = []
    rouge_scores: List[float] = []

    for idx, row in enumerate(rows):
        prompt = build_prompt(row["architecture"])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        generated = generated.split("Explanation:")[-1].strip()

        reference = row["explanation"].strip()

        bleu = compute_bleu(reference, generated)
        rouge_l = rouge.score(reference, generated)["rougeL"].fmeasure

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

        if idx == 0:
            print("Generated explanation:")
            print(generated)
            print("Ground truth:")
            print(reference)
            print("Difference:")
            diff = difflib.unified_diff(
                reference.splitlines(),
                generated.splitlines(),
                fromfile="ground_truth",
                tofile="generated",
                lineterm="",
            )
            for line in diff:
                print(line)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Evaluated {idx + 1}/{len(rows)} samples")

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    print(f"Average BLEU-4: {avg_bleu:.4f}")
    print(f"Average ROUGE-L: {avg_rouge:.4f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(rows),
                "average_bleu4": avg_bleu,
                "average_rougeL": avg_rouge,
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation report to {report_path}")
    print("[STEP 3/3] Evaluation completed")


if __name__ == "__main__":
    main()
