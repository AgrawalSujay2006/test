#!/usr/bin/env bash
# =============================================================================
# setup.bash — Run ONCE with internet to prepare everything
# =============================================================================

set -e   # exit immediately on any error

# ── CONFIG — UPDATE THESE BEFORE SUBMITTING ──────────────────
REPO_URL="https://github.com/AgrawalSujay2006/test.git"
ENV_NAME="gnr_project_env"
PYTHON_VER="3.11"
# ─────────────────────────────────────────────────────────────

echo "============================================================"
echo "  GNR Project — Environment Setup"
echo "============================================================"

# ── 1. Create Conda environment ───────────────────────────────
echo "[setup] Creating conda env: $ENV_NAME (Python $PYTHON_VER)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. System libraries ───────────────────────────────────────
echo "[setup] Installing system libraries..."
conda install -y -c conda-forge libgl glib ffmpeg

# ── 3. Python packages ────────────────────────────────────────
echo "[setup] Installing Python packages..."
pip install --upgrade pip

# PyTorch — cu121 wheel is compatible with CUDA 12.6 on L40s
pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

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

# ── 4. Clone project repo into CURRENT DIRECTORY ──────────────
echo "[setup] Cloning project repo into current directory..."
# We clone into a temp folder and move it so it doesn't crash 
# if the current directory isn't perfectly empty.
git clone "$REPO_URL" temp_repo
cp -r temp_repo/* .
rm -rf temp_repo

# ── 5. Download InternVL2-8B ──────────────────────────────────
echo "[setup] Downloading InternVL2-8B weights..."
mkdir -p models

python - <<'PYEOF'
import os, json
from huggingface_hub import snapshot_download

model_dir = "./models/InternVL2-8B"

if not os.path.isdir(model_dir) or not os.listdir(model_dir):
    print("  Downloading InternVL2-8B (~16 GB)...")
    snapshot_download(
        repo_id="OpenGVLab/InternVL2-8B",
        local_dir=model_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot"],
    )
    print("  Download complete.")
else:
    print("  InternVL2-8B already present, skipping download.")

# ── Patch configs to block any HuggingFace hub calls at runtime ──
for fname in ["config.json", "tokenizer_config.json"]:
    fpath = os.path.join(model_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            cfg = json.load(f)
        cfg.pop("auto_map", None)
        with open(fpath, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Patched {fname} (removed auto_map).")
PYEOF

# ── 6. Download LLaVA-1.5-7B (fallback) ──────────────────────
echo "[setup] Downloading LLaVA-1.5-7B weights (fallback)..."

python - <<'PYEOF'
import os, json
from huggingface_hub import snapshot_download

model_dir = "./models/llava-1.5-7b-hf"

if not os.path.isdir(model_dir) or not os.listdir(model_dir):
    print("  Downloading LLaVA-1.5-7B (~14 GB)...")
    snapshot_download(
        repo_id="llava-hf/llava-1.5-7b-hf",
        local_dir=model_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print("  Download complete.")
else:
    print("  LLaVA-1.5-7B already present, skipping download.")

for fname in ["config.json", "tokenizer_config.json"]:
    fpath = os.path.join(model_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            cfg = json.load(f)
        cfg.pop("auto_map", None)
        with open(fpath, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Patched {fname}.")
PYEOF

# ── 7. Bake TRANSFORMERS_OFFLINE into the conda env ──────────
echo "[setup] Baking offline env vars into conda activate..."

CONDA_BASE=$(conda info --base)
ACTIVATE_DIR="$CONDA_BASE/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

cat > "$ACTIVATE_DIR/set_offline.sh" <<'ENVEOF'
#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
ENVEOF

chmod +x "$ACTIVATE_DIR/set_offline.sh"
echo "  Offline vars will auto-set on 'conda activate $ENV_NAME'."

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
