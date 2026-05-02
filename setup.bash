#!/usr/bin/env bash
# =============================================================================
# setup.bash — Run ONCE with internet to prepare everything
# =============================================================================
# What this does:
#   1. Creates conda env  gnr_project_env  (Python 3.11)
#   2. Installs all Python dependencies
#   3. Clones the project repo
#   4. Downloads InternVL2-8B weights (primary VQA model)
#   5. Downloads LLaVA-1.5-7B weights (fallback VQA model)
#
# After this runs, everything works OFFLINE.
# =============================================================================

set -e   # exit immediately on any error

# ── CONFIG — UPDATE THESE ────────────────────────────────────────────────────
REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"   # ← CHANGE THIS
REPO_DIR="gnr_map_project"   # local folder name for the clone
ENV_NAME="gnr_project_env"
PYTHON_VER="3.11"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  GNR Project — Environment Setup"
echo "============================================================"

# ── 1. Create Conda environment ───────────────────────────────
echo "[setup] Creating conda env: $ENV_NAME (Python $PYTHON_VER)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# Activate inside script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Install system-level libraries (correct conda-forge names) ──
echo "[setup] Installing system libraries..."
conda install -y -c conda-forge \
    libgl \
    glib \
    ffmpeg

# ── 3. Install Python packages ────────────────────────────────
echo "[setup] Installing Python dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12.1 (compatible with CUDA 12.6 on L40s)
pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Core inference dependencies
pip install \
    transformers==4.43.4 \
    accelerate==0.33.0 \
    sentencepiece \
    einops \
    timm \
    Pillow \
    opencv-python-headless \
    numpy \
    pandas \
    tqdm \
    scipy \
    scikit-image \
    huggingface_hub \
    bitsandbytes

# ── 4. Clone project repo ─────────────────────────────────────
echo "[setup] Cloning project repo..."
cd ~
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── 5. Download model weights (into the project directory) ────
echo "[setup] Downloading InternVL2-8B weights..."
mkdir -p models

python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

model_dir = "./models/InternVL2-8B"
if not os.path.isdir(model_dir) or not os.listdir(model_dir):
    print("  Downloading InternVL2-8B ...")
    snapshot_download(
        repo_id="OpenGVLab/InternVL2-8B",
        local_dir=model_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot"]
    )
    print("  Done.")
else:
    print("  InternVL2-8B already present, skipping.")
PYEOF

echo "[setup] Downloading LLaVA-1.5-7B weights (fallback)..."
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

model_dir = "./models/llava-1.5-7b-hf"
if not os.path.isdir(model_dir) or not os.listdir(model_dir):
    print("  Downloading LLaVA-1.5-7B ...")
    snapshot_download(
        repo_id="llava-hf/llava-1.5-7b-hf",
        local_dir=model_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print("  Done.")
else:
    print("  LLaVA-1.5-7B already present, skipping.")
PYEOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To run inference:"
echo "    conda activate $ENV_NAME"
echo "    cd ~/$REPO_DIR"
echo "    python inference.py --test_dir <absolute_path_to_test_dir>"
echo "============================================================"