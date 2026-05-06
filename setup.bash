#!/usr/bin/env bash
# =============================================================================
# setup.bash — GNR Map Project  (FIXED)
# =============================================================================
set -e

REPO_URL="https://github.com/AgrawalSujay2006/test.git"
ENV_NAME="gnr_project_env"
PYTHON_VER="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  GNR Project — Environment Setup"
echo "  Project dir : $SCRIPT_DIR"
echo "============================================================"

# 1. Create conda env
echo "[setup] Creating conda env: $ENV_NAME (Python $PYTHON_VER)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2. System libraries
echo "[setup] Installing system libraries..."
conda install -y -c conda-forge libgl glib ffmpeg

# 3. Python packages
echo "[setup] Installing Python packages..."
pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install \
    transformers==4.43.4 accelerate==0.33.0 sentencepiece einops timm \
    Pillow opencv-python-headless numpy pandas tqdm scipy scikit-image \
    huggingface_hub bitsandbytes

# 4. Clone repo — copy files into SCRIPT_DIR
echo "[setup] Cloning repo..."
TMP="$(mktemp -d)"
git clone "$REPO_URL" "$TMP/repo"
cp "$TMP/repo/inference.py" "$SCRIPT_DIR/inference.py"
cp "$TMP/repo/stitcher.py"  "$SCRIPT_DIR/stitcher.py"
rm -rf "$TMP"
echo "[setup] inference.py and stitcher.py copied to $SCRIPT_DIR"

# 5. Download model weights into SCRIPT_DIR/models
echo "[setup] Downloading InternVL2-8B weights..."
cd "$SCRIPT_DIR"
mkdir -p models

python3 - <<'PYEOF'
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
    print("  Done.")
else:
    print("  Already present, skipping.")

# Strip auto_map from config.json to block hub calls.
# inference.py restores it at runtime.
cpath = os.path.join(model_dir, "config.json")
if os.path.exists(cpath):
    with open(cpath) as f: cfg = json.load(f)
    cfg.pop("auto_map", None)
    with open(cpath, "w") as f: json.dump(cfg, f, indent=2)
    print("  config.json patched.")

# KEY FIX: remove BOTH auto_map AND tokenizer_class from tokenizer_config.json.
#
# Root cause of TA's error:
#   "ValueError: Tokenizer class InternLM2Tokenizer does not exist"
#
# The old code only removed auto_map but left:
#   "tokenizer_class": "InternLM2Tokenizer"
#
# Without auto_map, transformers tries to look up "InternLM2Tokenizer"
# in its built-in registry (TOKEN_MAPPING). InternLM2Tokenizer is a
# *custom* class shipped in the model files — not registered there.
# Result: ValueError crash every time.
#
# Fix: remove tokenizer_class too. inference.py restores auto_map
# at runtime, so transformers uses the local Python file instead.
tpath = os.path.join(model_dir, "tokenizer_config.json")
if os.path.exists(tpath):
    with open(tpath) as f: tcfg = json.load(f)
    tcfg.pop("auto_map", None)
    tcfg.pop("tokenizer_class", None)   # <-- THIS FIXES THE TA'S ERROR
    with open(tpath, "w") as f: json.dump(tcfg, f, indent=2)
    print("  tokenizer_config.json patched (tokenizer_class removed).")
PYEOF

# 6. Bake offline env vars into conda activate
echo "[setup] Baking offline vars into conda activate..."
CONDA_BASE=$(conda info --base)
ACTIVATE_DIR="$CONDA_BASE/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/set_offline.sh" << 'ENVEOF'
#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
ENVEOF
chmod +x "$ACTIVATE_DIR/set_offline.sh"

# 7. Verify layout
echo ""
echo "[setup] Verifying layout..."
MISSING=0
chk() { [ -e "$1" ] && echo "  OK : $1" || { echo "  MISSING : $1"; MISSING=$((MISSING+1)); }; }
chk "$SCRIPT_DIR/inference.py"
chk "$SCRIPT_DIR/stitcher.py"
chk "$SCRIPT_DIR/models/InternVL2-8B/config.json"
chk "$SCRIPT_DIR/models/InternVL2-8B/tokenizer_config.json"
chk "$SCRIPT_DIR/models/InternVL2-8B/tokenization_internlm2.py"
[ "$MISSING" -gt 0 ] && echo "  WARNING: $MISSING file(s) missing" || echo "  All files present."

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  conda activate $ENV_NAME"
echo "  python inference.py --test_dir <absolute_path_to_test_dir>"
echo "============================================================"
