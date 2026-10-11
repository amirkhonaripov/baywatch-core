#!/usr/bin/env bash
# Create .venv with everything BayWatch needs (Linux + NVIDIA GPU, Python 3.12). Uses uv (https://docs.astral.sh/uv/).
#   scripts/setup_env.sh            # inference + API
#   scripts/setup_env.sh --eval     # also the evaluation / training extras
# TORCH_CUDA=cu128 scripts/setup_env.sh   on drivers older than 580.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu130}" torch==2.14.1 torchvision==0.29.1
uv pip install -r requirements.txt
[[ "${1:-}" == "--eval" ]] && uv pip install -r requirements-eval.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "next: source .venv/bin/activate && python scripts/download_weights.py"
