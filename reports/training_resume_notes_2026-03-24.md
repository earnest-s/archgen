# ArchitectAI Training Resume Notes (2026-03-24)

Status snapshot:
- Root virtual environment exists at .venv
- Dependencies installed successfully inside .venv
- Core imports verified: torch, timm, transformers, peft, bitsandbytes
- Current torch CUDA state before driver fix: cuda=False
- Full training command was started but intentionally canceled by user

Runtime issue encountered and resolved:
- pip install originally failed with "No space left on device" during CUDA wheel download
- Re-ran install using no cache and workspace temp directory, then dependencies installed successfully

Code updates applied (minimal, startup/runtime focused only):
1) scripts/run_training.py
- Fixed Stage 2 data argument bug (pass data directory, not dataset.jsonl path)
- Aligned default ConvNeXt output to checkpoints (so convnext_best.pt lands at checkpoints/convnext_best.pt)
- Updated checkpoint verification to check:
  - checkpoints/convnext_best.pt
  - checkpoints/qwen_lora/
- Added stage progress prints (1/2 and 2/2)
- Added GPU usage print helper (reports CUDA availability/memory when available)
- Added clear crash handling with traceback and proper abort summary
- Added stop-on-failure for Stage 2 as requested

2) backend/training/vision/train_convnext.py
- Added optional checkpoint preload if checkpoints/convnext_best.pt already exists
- Keeps timm.create_model("convnext_tiny", pretrained=True) behavior intact
- Model is still moved to CUDA if available via existing loader logic

Resume checklist after NVIDIA driver fix:
1. Verify GPU driver:
- nvidia-smi

2. Verify CUDA from project venv:
- .venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

3. Run full pipeline:
- .venv/bin/python scripts/run_training.py

4. Verify required outputs:
- ls -lah checkpoints/convnext_best.pt
- ls -lah checkpoints/qwen_lora/

Notes:
- Qwen loader uses 4-bit quantization with BitsAndBytesConfig and device_map="auto"
- LoRA setup includes gradient_checkpointing and model.enable_input_require_grads()
- Model download caching is handled by Hugging Face local cache
