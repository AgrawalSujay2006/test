import os
# Force offline mode for all libraries
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import argparse, re, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Ensure stitcher.py is in your repo/folder
from stitcher import stitch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True, help="Directory containing test.csv and patches")
    p.add_argument("--strategy", default="auto")
    return p.parse_args()

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def prepare_map_image(bgr, max_dim=1344):
    h, w  = bgr.shape[:2]
    scale = min(max_dim/w, max_dim/h, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_AREA)
        print(f"[img] Resized to {bgr.shape[1]}x{bgr.shape[0]} px")
    return bgr_to_pil(bgr)

def load_internvl(model_dir):
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    print(f"[vqa] Loading InternVL2 from {model_dir}...")
    
    # Check GPU memory to decide on quantization
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        trust_remote_code=True,
        use_fast=False, 
        local_files_only=True
    )
    
    if vram_gb < 20:
        print(f"[vqa] VRAM: {vram_gb:.1f}GB — Using 4-bit quantization")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModel.from_pretrained(
            model_dir, 
            quantization_config=bnb,
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            local_files_only=True
        ).eval()
    else:
        print(f"[vqa] VRAM: {vram_gb:.1f}GB — Using bfloat16")
        model = AutoModel.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            local_files_only=True
        ).eval().cuda()
            
    return model, tokenizer

def load_image_for_internvl(image):
    """Specific preprocessing for InternVL2 8B"""
    target_size = 448
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Expand to 2x2 grid + 1 global view
    resized_img = image.resize((target_size * 2, target_size * 2))
    images = []
    for i in range(4):
        x, y = (i % 2) * target_size, (i // 2) * target_size
        images.append(resized_img.crop((x, y, x + target_size, y + target_size)))
    
    images.append(image.resize((target_size, target_size)))
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

def internvl_answer(model, tokenizer, pil_image, question, options):
    prompt = (
        "You are a geographic expert analyzing a detailed map image.\n\n"
        f"Question: {question}\n"
        f"Options:\n1. {options[0]}\n2. {options[1]}\n"
        f"3. {options[2]}\n4. {options[3]}\n\n"
        "Carefully examine the map. Reply with ONLY a single digit: 1, 2, 3, or 4. "
        "If you cannot determine the answer, reply 5."
    )
    
    pixel_values = load_image_for_internvl(pil_image)
    compute_dtype = torch.float16 if model.dtype == torch.float32 else model.dtype
    pixel_values = pixel_values.to(model.device, dtype=compute_dtype)
    
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt,
                              dict(max_new_tokens=8, do_sample=False))
                          
    digits = re.findall(r"[1-5]", str(response))
    return int(digits[0]) if digits else 5

def load_vqa_model(script_dir):
    # Search paths for the weights downloaded in setup.bash
    paths_to_check = [
        os.path.join(script_dir, "model_weights"),
        os.path.join(script_dir, "..", "model_weights"),
        os.path.join(script_dir, "models", "InternVL2-8B")
    ]
    
    for mdir in paths_to_check:
        if os.path.isdir(mdir) and any(f.endswith('.bin') or f.endswith('.safetensors') for f in os.listdir(mdir)):
            try:
                m, p = load_internvl(mdir)
                return m, p, internvl_answer
            except Exception as e:
                print(f"[vqa] Failed loading from {mdir}: {e}")
                
    print("[WARN] No weights found offline! Defaulting to 'unsure' (5).")
    return None, None, None

def main():
    args       = parse_args()
    test_dir   = Path(args.test_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    output_csv = Path("submission.csv")

    # 1. Stitch patches
    # Check if 'patches' exists as a subfolder, otherwise use test_dir
    patches_path = test_dir / "patches"
    if not patches_path.exists():
        patches_path = test_dir
        
    cache = str(script_dir / "stitched_map.png")
    print(f"[step] Stitching patches from {patches_path}...")
    stitched = stitch(str(patches_path), cache, strategy=args.strategy)
    
    # 2. Prepare image for VLM
    pil_map = prepare_map_image(stitched, max_dim=1344)
    
    # 3. Load VQA Model
    vqa_model, vqa_proc, vqa_fn = load_vqa_model(str(script_dir))

    # 4. Process CSV
    test_csv = test_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test.csv in {test_dir}")
    
    test_df = pd.read_csv(test_csv)
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        qid = str(row["id"])
        try:
            opts = [str(row[f"option_{i}"]) for i in range(1, 5)]
            if vqa_model:
                ans = vqa_fn(vqa_model, vqa_proc, pil_map, str(row["question"]), opts)
            else:
                ans = 5
        except Exception as e:
            print(f" Error on QID {qid}: {e}")
            ans = 5
            
        results.append({"id": qid, "question_num": qid, "option": ans})

    # 5. Save Results
    sub = pd.DataFrame(results)
    sub.to_csv(output_csv, index=False)
    print(f"\n[done] Saved {len(sub)} answers to {output_csv}")

if __name__ == "__main__":
    main()
