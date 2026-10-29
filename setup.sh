#!/usr/bin/env bash
#
# Environment setup with uv (https://docs.astral.sh/uv/).
#
# Install uv first if you don't have it:
#     curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Then, from the repo root:
#     ./setup.sh
#
# This creates a local `.venv` and installs the dependencies into it.
# Activate it for subsequent runs with:  source .venv/bin/activate
set -euo pipefail

# Create the virtual environment (uv defaults to ./.venv).
uv venv --python 3.11

# `uv pip` is a fast drop-in for pip; it targets the .venv created above.
# NOTE: the experiments run against the *vendored* TransformerLens in
# third_party/ (shared.paths.setup() puts it ahead of the pip copy on
# sys.path); the pip install here is only for its dependencies.
uv pip install \
  "transformers==5.21.0" tokenizers accelerate \
  "transformer-lens==4.3.0" "better-abc==0.0.3" \
  torch torchvision \
  qwen-vl-utils datasets \
  matplotlib pillow tqdm ipykernel \
  huggingface_hub

echo
echo "Dependencies installed into .venv — activate with: source .venv/bin/activate"
echo
echo "For gated models (Qwen2-VL, Pixtral, ...) authenticate with Hugging Face:"
echo "    uv run huggingface-cli login        # interactive, or"
echo "    export HF_TOKEN=hf_xxx               # non-interactive (read from .env by the runners)"
