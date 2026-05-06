import os, sys

# ══════════════════════════════════════════════════════════════
# Set offline flags FIRST — before any import that touches
# transformers. If set later, the hub client may already be
# initialised and ignore the flag.
# ══════════════════════════════════════════════════════════════
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import re, json, warnings, traceback
warnings.filterwarnings("ignore")

import cv2, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from stitcher import stitch


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True,
                   help="Absolute path to test directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────
def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
])


def make_tiles(pil, max_dim=1344):
    """5 overlapping tiles so the model can read small map labels."""
    def rsz(img):
        w, h  = img.size
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)),
                             Image.Resampling.LANCZOS)
        return img.convert("RGB")

    w, h   = pil.size
    cx, cy = w // 2, h // 2
    ox, oy = int(w * 0.10), int(h * 0.10)
    return [
        rsz(pil),
        rsz(pil.crop((0,     0,     cx+ox, cy+oy))),
        rsz(pil.crop((cx-ox, 0,     w,     cy+oy))),
        rsz(pil.crop((0,     cy-oy, cx+ox, h    ))),
        rsz(pil.crop((cx-ox, cy-oy, w,     h    ))),
    ]


def pil_to_pixel_values(pil_img, model):
    """
    PIL Image -> (1, C, H, W) tensor matching the loaded model's
    dtype and device. Works for bfloat16, float16, and 4-bit.
    """
    try:
        param  = next(model.vision_model.parameters())
        device, dtype = param.device, param.dtype
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.bfloat16
    return _transform(pil_img).unsqueeze(0).to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────
# PROMPT / ANSWER PARSING
# ─────────────────────────────────────────────────────────────
def build_prompt(question, options):
    opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
    return (
        "You are a geographic expert analyzing a map image.\n\n"
        f"Question: {question}\n\nOptions:\n{opts}\n\n"
        "Carefully examine all text labels, roads, water bodies, "
        "and landmarks visible in the map.\n"
        "End your response with exactly:  ANSWER: <digit>\n"
        "where <digit> is 1, 2, 3, or 4. "
        "If you cannot determine the answer write:  ANSWER: 5\n"
    )


def extract_answer(text):
    m = re.search(r"ANSWER\s*[:=]\s*([1-5])", str(text), re.IGNORECASE)
    if m:
        return int(m.group(1))
    digits = re.findall(r"[1-5]", str(text))
    return int(digits[-1]) if digits else 5


# ─────────────────────────────────────────────────────────────
# MODEL LOADING  — THE ACTUAL FIX IS HERE
# ─────────────────────────────────────────────────────────────
def load_internvl(model_dir):
    """
    WHY THE TA GETS:
        ValueError: Tokenizer class InternLM2Tokenizer does not exist

    setup.bash removes auto_map from tokenizer_config.json (to block
    hub calls at inference time). Correct goal, wrong execution.

    tokenizer_config.json is left with:
        { "tokenizer_class": "InternLM2Tokenizer", ... }   <-- no auto_map

    When transformers sees tokenizer_class without auto_map it looks up
    "InternLM2Tokenizer" in its internal TOKEN_MAPPING (built-in classes
    only). InternLM2Tokenizer is a *custom* class living in the model's
    local Python files — it is not registered there. Crash.

    FIX: at runtime, before from_pretrained(), restore auto_map in
    tokenizer_config.json AND remove tokenizer_class so transformers
    uses auto_map (local file lookup) instead of the registry.

    No internet needed — the tokenization_internlm2.py file was already
    downloaded into model_dir by setup.bash.
    """
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    # 1. Patch config.json  (model architecture)
    cpath = os.path.join(model_dir, "config.json")
    if os.path.exists(cpath):
        with open(cpath) as f: cfg = json.load(f)
        cfg["auto_map"] = {
            "AutoConfig":           "configuration_internvl_chat.InternVLChatConfig",
            "AutoModel":            "modeling_internvl_chat.InternVLChatModel",
            "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel",
        }
        with open(cpath, "w") as f: json.dump(cfg, f, indent=2)
        print("[load] config.json  — auto_map restored")

    # 2. THE KEY FIX: patch tokenizer_config.json
    tpath = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tpath):
        with open(tpath) as f: tcfg = json.load(f)
        tcfg["auto_map"] = {
            "AutoTokenizer": [
                "tokenization_internlm2.InternLM2Tokenizer",
                "tokenization_internlm2_fast.InternLM2TokenizerFast",
            ]
        }
        tcfg.pop("tokenizer_class", None)   # remove registry lookup
        with open(tpath, "w") as f: json.dump(tcfg, f, indent=2)
        print("[load] tokenizer_config.json — auto_map restored, "
              "tokenizer_class removed")

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True,
        use_fast=False, local_files_only=True)
    print("[load] Tokenizer OK")

    # 4. Load model  (full bf16 on L40s, 4-bit on smaller GPUs)
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0)
    print(f"[load] VRAM: {vram:.1f} GB")

    if vram >= 20:
        model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
            local_files_only=True).eval().cuda()
        print("[load] bfloat16 model loaded")
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModel.from_pretrained(
            model_dir, quantization_config=bnb,
            low_cpu_mem_usage=True, trust_remote_code=True,
            local_files_only=True).eval()
        print("[load] 4-bit model loaded")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# VQA
# ─────────────────────────────────────────────────────────────
def internvl_answer(model, tokenizer, tiles, question, options):
    from collections import Counter
    prompt  = build_prompt(question, options)
    gen_cfg = dict(max_new_tokens=64, do_sample=False)
    answers = []

    for i, tile_pil in enumerate(tiles):
        try:
            pv       = pil_to_pixel_values(tile_pil, model)
            response = model.chat(tokenizer, pv, prompt, gen_cfg)
            ans      = extract_answer(response)
            answers.append(ans)
            print(f"    tile {i} [{ans}] {str(response)[:80]}")
        except Exception as e:
            print(f"    tile {i} ERROR: {type(e).__name__}: {e}")

    counts = Counter(a for a in answers if a != 5)
    return counts.most_common(1)[0][0] if counts else 5


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    test_dir   = Path(args.test_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    output_csv = Path("submission.csv")           # CWD per grading spec
    cache_path = str(script_dir / "stitched_map.png")
    model_dir  = str(script_dir / "models" / "InternVL2-8B")

    print("=" * 60)
    print("  GNR MAP INFERENCE")
    print(f"  test_dir   : {test_dir}")
    print(f"  script_dir : {script_dir}")
    print(f"  model_dir  : {model_dir}")
    print(f"  output     : {output_csv.resolve()}")
    print("=" * 60)

    # 1. Stitch
    patches_path = test_dir / "patches"
    if not patches_path.exists():
        patches_path = test_dir
    try:
        stitched = stitch(str(patches_path), cache_path)
    except Exception as e:
        print(f"[WARN] Stitcher failed: {e} — using patch_0 fallback")
        fb = str(patches_path / "patch_0.png")
        stitched = cv2.imread(fb)
        if stitched is None:
            print("[ERROR] patch_0.png missing. Cannot continue.")
            sys.exit(1)
        cv2.imwrite(cache_path, stitched)

    tiles = make_tiles(bgr_to_pil(stitched))
    print(f"[img] {len(tiles)} tiles from {Path(cache_path).name}")

    # 2. Load model
    if not os.path.isdir(model_dir):
        print(f"[ERROR] Model missing: {model_dir}")
        print("        Re-run setup.bash to download weights.")
        sys.exit(1)
    model, tokenizer = load_internvl(model_dir)

    # 3. Load questions
    test_csv = test_dir / "test.csv"
    if not test_csv.exists():
        print(f"[ERROR] test.csv not found: {test_csv}")
        sys.exit(1)
    test_df = pd.read_csv(test_csv)
    print(f"[qa] {len(test_df)} questions")

    # 4. Answer
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="Inference"):
        qid = str(row["id"])
        try:
            opts = [str(row[f"option_{i}"]) for i in range(1, 5)]
            opts = [o for o in opts if o not in ("nan", "None", "")]
            print(f"\n[{qid}] {row['question']}")
            ans = internvl_answer(model, tokenizer, tiles,
                                  str(row["question"]), opts)
        except Exception as e:
            print(f"  [{qid}] EXCEPTION: {e}")
            traceback.print_exc()
            ans = 5
        results.append({"id": qid, "question_num": qid, "option": ans})
        print(f"  [{qid}] -> {ans}")

    # 5. Save
    sub = pd.DataFrame(results, columns=["id", "question_num", "option"])
    sub.to_csv(str(output_csv), index=False)
    print(f"\n[done] {len(sub)} rows -> {output_csv.resolve()}")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
