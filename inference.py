"""
inference.py — Map Reconstruction & MCQ Answering (Offline)
============================================================
Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

Outputs:
    ./submission.csv  (in the current working directory, NOT in test_dir)

Pipeline:
    1. Load and stitch map patches from test_dir/patches/
    2. Load questions from test_dir/test.csv
    3. Run InternVL2-8B (or LLaVA-1.5-7B fallback) for VQA
    4. Write submission.csv
"""

import argparse
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Local module
from stitcher import stitch


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline map stitching + MCQ answering")
    parser.add_argument("--test_dir", required=True,
                        help="Absolute path to test directory")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "sift", "grid", "opencv"],
                        help="Stitching strategy (default: auto)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def prepare_map_image(stitched_bgr: np.ndarray,
                      max_dim: int = 1344) -> Image.Image:
    """
    Resize the stitched map so the longest side ≤ max_dim.
    Returns a PIL Image (RGB).
    """
    h, w = stitched_bgr.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        stitched_bgr = cv2.resize(stitched_bgr, (nw, nh),
                                  interpolation=cv2.INTER_AREA)
        print(f"[img] Resized map to {nw}×{nh} px")
    return bgr_to_pil(stitched_bgr)


# ─────────────────────────────────────────────────────────────
# VQA — InternVL2-8B
# ─────────────────────────────────────────────────────────────

def load_internvl(model_dir: str):
    """Load InternVL2-8B from local path."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(f"[vqa] Loading InternVL2-8B from {model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    print("[vqa] InternVL2-8B loaded.")
    return model, tokenizer


def internvl_answer(model, tokenizer,
                    pil_image: Image.Image,
                    question: str,
                    options: list) -> int:
    """
    Ask InternVL2 a MCQ question. Returns 1-4, or 5 if unsure.
    """
    prompt = (
        "You are a geographic expert analyzing a detailed map image.\n\n"
        f"Question: {question}\n"
        f"Options:\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n\n"
        "Carefully examine the map and choose the correct option.\n"
        "Reply with ONLY a single digit: 1, 2, 3, or 4.\n"
        "If you genuinely cannot determine the answer, reply 5."
    )

    generation_config = dict(max_new_tokens=8, do_sample=False)

    # InternVL2 expects max ~448×448 or tiled 1024
    pil_resized = pil_image.resize((1024, 1024), Image.LANCZOS)

    response = model.chat(
        tokenizer, pil_resized, prompt,
        generation_config=generation_config
    )

    digits = re.findall(r"[1-5]", str(response))
    return int(digits[0]) if digits else 5


# ─────────────────────────────────────────────────────────────
# VQA — LLaVA-1.5-7B (fallback)
# ─────────────────────────────────────────────────────────────

def load_llava(model_dir: str):
    """Load LLaVA-1.5-7B from local path."""
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
    import torch

    print(f"[vqa] Loading LLaVA-1.5-7B from {model_dir} …")
    processor = LlavaProcessor.from_pretrained(model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    print("[vqa] LLaVA-1.5-7B loaded.")
    return model, processor


def llava_answer(model, processor,
                 pil_image: Image.Image,
                 question: str,
                 options: list) -> int:
    """
    Ask LLaVA a MCQ question. Returns 1-4, or 5 if unsure.
    """
    import torch

    prompt_text = (
        "USER: <image>\n"
        "You are analyzing a geographic map image.\n"
        f"Question: {question}\n"
        "Options:\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n"
        "Reply with ONLY one digit: 1, 2, 3, 4, or 5 if unsure.\n"
        "ASSISTANT:"
    )

    inputs = processor(text=prompt_text, images=pil_image, return_tensors="pt")
    inputs = {k: v.cuda() if hasattr(v, "cuda") else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=8, do_sample=False)

    decoded = processor.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)

    digits = re.findall(r"[1-5]", decoded)
    return int(digits[0]) if digits else 5


# ─────────────────────────────────────────────────────────────
# MODEL LOADER (tries InternVL then LLaVA)
# ─────────────────────────────────────────────────────────────

def load_vqa_model(script_dir: str):
    """
    Try to load InternVL2-8B, then LLaVA-1.5-7B.
    Returns (model, processor_or_tokenizer, answer_fn) or (None, None, None).
    model_dir is resolved relative to the directory of inference.py.
    """
    internvl_dir = os.path.join(script_dir, "models", "InternVL2-8B")
    llava_dir    = os.path.join(script_dir, "models", "llava-1.5-7b-hf")

    if os.path.isdir(internvl_dir) and os.listdir(internvl_dir):
        try:
            m, p = load_internvl(internvl_dir)
            return m, p, internvl_answer
        except Exception as e:
            print(f"[vqa] InternVL2 load failed: {e}")

    if os.path.isdir(llava_dir) and os.listdir(llava_dir):
        try:
            m, p = load_llava(llava_dir)
            return m, p, llava_answer
        except Exception as e:
            print(f"[vqa] LLaVA load failed: {e}")

    print("[WARN] No VQA model found. All answers will be 5 (unanswered).")
    return None, None, None


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    test_dir    = Path(args.test_dir).resolve()
    patches_dir = test_dir / "patches"
    test_csv    = test_dir / "test.csv"
    output_csv  = Path("submission.csv")   # always in CWD (project dir)

    # script_dir is the directory containing inference.py (= project dir)
    script_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("  MAP RECONSTRUCTION & MCQ — OFFLINE INFERENCE")
    print("=" * 60)
    print(f"  test_dir    : {test_dir}")
    print(f"  patches_dir : {patches_dir}")
    print(f"  test_csv    : {test_csv}")
    print(f"  output_csv  : {output_csv.resolve()}")
    print(f"  script_dir  : {script_dir}")
    print("=" * 60)

    # ── 1. Stitch map ────────────────────────────────────────
    cache_path = str(script_dir / "stitched_map.png")
    stitched_bgr = stitch(
        patches_dir=str(patches_dir),
        output_path=cache_path,
        strategy=args.strategy,
    )
    pil_map = prepare_map_image(stitched_bgr, max_dim=1344)
    print(f"[img] PIL map size: {pil_map.size}")

    # Also save resized version for debugging
    pil_map.save(str(script_dir / "stitched_map_resized.jpg"), quality=90)

    # ── 2. Load questions ────────────────────────────────────
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"[qa] {len(test_df)} questions loaded")

    # ── 3. Load VQA model ────────────────────────────────────
    vqa_model, vqa_proc, vqa_fn = load_vqa_model(str(script_dir))

    # ── 4. Answer questions ──────────────────────────────────
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="Answering"):
        qid = str(row["id"])
        try:
            options = [
                str(row["option_1"]),
                str(row["option_2"]),
                str(row["option_3"]),
                str(row["option_4"]),
            ]
            if vqa_model is not None:
                ans = vqa_fn(vqa_model, vqa_proc, pil_map,
                             str(row["question"]), options)
            else:
                ans = 5   # unanswered — no penalty
        except Exception as e:
            print(f"  [error] {qid}: {e}")
            ans = 5

        results.append({"id": qid, "question_num": qid, "option": ans})
        print(f"  {qid} → {ans}")

    # ── 5. Write submission ──────────────────────────────────
    sub = pd.DataFrame(results, columns=["id", "question_num", "option"])
    sub.to_csv(str(output_csv), index=False)
    print(f"\n[done] {len(sub)} rows saved → {output_csv.resolve()}")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()