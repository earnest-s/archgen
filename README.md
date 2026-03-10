# ArchitectAI

A deep learning-assisted tool that generates and explains software architecture diagrams from natural-language prompts.

## Overview

ArchitectAI converts English architecture descriptions into structured diagrams and AI-generated explanations. Diagrams are programmatically generated and then interpreted by vision-language models. The system runs locally on GPU.

## Core Capabilities

- Natural-language architecture input
- Automatic architecture diagram generation
- Editable structured diagrams
- Vision-based diagram understanding
- AI-generated architecture explanation
- Local deployment

## System Pipeline

Prompt -> Architecture JSON -> Diagram PNG -> Vision Encoder -> Language Model -> Explanation

This pipeline includes an optional user editing loop to refine structured diagrams before explanation.

## Tech Stack (Current)

- Vision: ConvNeXt-Tiny (PyTorch, timm)
- Language: Qwen2.5-3B-Instruct (4-bit, Transformers, PEFT, bitsandbytes)
- Diagram: diagrams + Graphviz
- Backend: FastAPI
- Frontend: React + TypeScript + React Flow
- Deployment: Docker (CUDA, WSL2)

## Repository Structure

The repository is structured to separate backend, frontend, models, and shared schema directories to maintain modular development.

## Status

The project is under active development and components are being implemented incrementally.