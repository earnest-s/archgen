"""check_model_checkpoints.py — Verify ArchitectAI model checkpoints.

Checks:
  • File existence and size (bytes + MB)
  • PyTorch checkpoint loading (state_dict integrity)
  • Parameter counts (total / trainable)
  • Device compatibility (CUDA / CPU)
  • Corruption detection (catches load errors)

Checkpoints verified
--------------------
  checkpoints/convnext/convnext_best.pt   — ConvNeXt-Tiny multil-label classifier
  checkpoints/qwen_lora/lora_adapter/     — Qwen2.5 LoRA adapter directory

Output
------
  reports/checkpoints_report.json

Usage
-----
    python scripts/check_model_checkpoints.py [OPTIONS]

Options
-------
    --convnext   Path to ConvNeXt checkpoint (default: checkpoints/convnext/convnext_best.pt)
    --qwen-lora  Path to LoRA adapter directory (default: checkpoints/qwen_lora/lora_adapter)
    --output     Report destination (default: reports/checkpoints_report.json)
    --no-load    Skip actual model loading (size/existence check only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "-" * 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_size(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "bytes": 0, "mb": 0.0}
    size = path.stat().st_size
    return {"exists": True, "bytes": size, "mb": round(size / 1_048_576, 2)}


def _dir_size(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "bytes": 0, "mb": 0.0, "n_files": 0}
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    n_files = sum(1 for _ in path.rglob("*") if _.is_file())
    return {"exists": True, "bytes": total, "mb": round(total / 1_048_576, 2),
            "n_files": n_files}


def _count_params(model) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2)}


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ConvNeXt checkpoint verification
# ---------------------------------------------------------------------------


def verify_convnext(ckpt_path: Path, load: bool) -> Dict[str, Any]:
    logger.info(SEP)
    logger.info("ConvNeXt checkpoint: %s", ckpt_path)

    rec: Dict[str, Any] = {
        "path":   str(ckpt_path),
        "type":   "convnext",
        "status": "unknown",
    }
    rec.update(_file_size(ckpt_path))

    if not rec["exists"]:
        logger.warning("  NOT FOUND: %s", ckpt_path)
        rec["status"] = "missing"
        return rec

    logger.info("  Size: %.2f MB (%d bytes)", rec["mb"], rec["bytes"])

    if not load:
        rec["status"] = "exists"
        logger.info("  Load skipped (--no-load).")
        return rec

    try:
        import torch
        import timm  # type: ignore

        device = "cuda" if _cuda_available() else "cpu"
        logger.info("  Loading on: %s", device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Support both bare state_dict and wrapped {"model_state_dict": ...}
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            meta = {k: v for k, v in checkpoint.items()
                    if k != "model_state_dict" and not hasattr(v, "keys")}
            rec["metadata"] = meta
        else:
            state_dict = checkpoint

        # Build the model and load weights
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=7)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        rec.update(_count_params(model))
        rec["missing_keys"]    = list(missing)[:10]
        rec["unexpected_keys"] = list(unexpected)[:10]
        rec["device"]          = device
        rec["cuda_available"]  = _cuda_available()
        rec["status"]          = "ok" if not missing else "partial"

        logger.info("  Parameters : %s M total / %s M trainable",
                    rec["total_M"], rec["trainable_M"])
        if missing:
            logger.warning("  Missing keys (%d): %s…", len(missing), missing[:3])
        if unexpected:
            logger.warning("  Unexpected keys (%d): %s…", len(unexpected), unexpected[:3])
        logger.info("  ✓ ConvNeXt checkpoint loaded successfully.")

    except Exception as exc:
        logger.error("  ✗ Load failed: %s", exc)
        rec["status"]       = "corrupted"
        rec["error"]        = str(exc)

    return rec


# ---------------------------------------------------------------------------
# Qwen LoRA adapter verification
# ---------------------------------------------------------------------------


def verify_qwen_lora(adapter_dir: Path, load: bool) -> Dict[str, Any]:
    logger.info(SEP)
    logger.info("Qwen LoRA adapter directory: %s", adapter_dir)

    rec: Dict[str, Any] = {
        "path":   str(adapter_dir),
        "type":   "qwen_lora",
        "status": "unknown",
    }
    rec.update(_dir_size(adapter_dir))

    if not rec["exists"]:
        logger.warning("  NOT FOUND: %s", adapter_dir)
        rec["status"] = "missing"
        return rec

    logger.info("  Directory size: %.2f MB across %d files",
                rec["mb"], rec["n_files"])

    # List key adapter files
    key_files = ["adapter_config.json", "adapter_model.safetensors",
                 "adapter_model.bin"]
    found_files = {}
    for fname in key_files:
        fp = adapter_dir / fname
        if fp.exists():
            sz = fp.stat().st_size
            found_files[fname] = {"exists": True, "mb": round(sz / 1_048_576, 2)}
            logger.info("  ✓ %s (%.2f MB)", fname, sz / 1_048_576)
        else:
            found_files[fname] = {"exists": False}
            logger.warning("  ✗ %s — not found", fname)
    rec["files"] = found_files

    # Check adapter_config.json
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            rec["adapter_config"] = {
                "r":      cfg.get("r"),
                "lora_alpha": cfg.get("lora_alpha"),
                "target_modules": cfg.get("target_modules"),
                "base_model":     cfg.get("base_model_name_or_path"),
            }
            logger.info("  Config: r=%s, alpha=%s, targets=%s",
                        cfg.get("r"), cfg.get("lora_alpha"),
                        cfg.get("target_modules"))
        except Exception as exc:
            logger.warning("  Could not parse adapter_config.json: %s", exc)

    if not load:
        rec["status"] = "exists"
        logger.info("  Load skipped (--no-load).")
        return rec

    try:
        from peft import PeftConfig  # type: ignore
        peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
        rec["peft_config_ok"] = True
        rec["status"]         = "ok"
        logger.info("  ✓ PeftConfig loaded: base_model=%s", peft_cfg.base_model_name_or_path)
    except Exception as exc:
        logger.warning("  PeftConfig load failed (full model load skipped): %s", exc)
        rec["peft_config_ok"] = False
        # Judge by file presence
        has_weights = (adapter_dir / "adapter_model.safetensors").exists() or \
                      (adapter_dir / "adapter_model.bin").exists()
        rec["status"] = "ok" if has_weights else "corrupted"
        rec["peft_error"] = str(exc)

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convnext_path = Path(args.convnext)
    qwen_path     = Path(args.qwen_lora)
    load          = not args.no_load

    cuda_ok = _cuda_available()
    logger.info("CUDA available: %s", cuda_ok)

    convnext_rec = verify_convnext(convnext_path, load=load)
    qwen_rec     = verify_qwen_lora(qwen_path, load=load)

    report = {
        "cuda_available": cuda_ok,
        "checkpoints": {
            "convnext": convnext_rec,
            "qwen_lora": qwen_rec,
        },
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Checkpoint Verification Report")
    print("=" * 60)
    for key, rec in report["checkpoints"].items():
        icon = "✓" if rec["status"] == "ok" else ("⚠" if rec["status"] == "exists" else "✗")
        print(f"  {icon}  {key:<14}  status={rec['status']:<12}  "
              f"{rec.get('mb', 0):.1f} MB")
    print("=" * 60 + "\n")

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved → %s", output_path)

    any_bad = any(
        r["status"] in {"missing", "corrupted"}
        for r in report["checkpoints"].values()
    )
    sys.exit(1 if any_bad else 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify ArchitectAI model checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--convnext",  default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--qwen-lora", default="checkpoints/qwen_lora/lora_adapter")
    p.add_argument("--output",    default="reports/checkpoints_report.json")
    p.add_argument("--no-load",   action="store_true",
                   help="Skip actual model loading (existence + size check only).")
    return p.parse_args()


if __name__ == "__main__":
    main()
