import os, re, warnings, traceback
warnings.filterwarnings("ignore")
import cv2, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from stitcher import stitch   

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True, help="Absolute path to test directory")
    return p.parse_args()

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def pil_to_pixel_values(pil_img, model):
    """FIX: Dynamically matches model dtype to prevent RuntimeError on T4/L40s."""
    try:
        param = next(model.vision_model.parameters())
        device, dtype = param.device, param.dtype 
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

    tensor = _transform(pil_img).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor

def make_tiles(pil, max_dim=1344):
    def rsz(img):
        w, h  = img.size
        scale = min(max_dim/w, max_dim/h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        return img.convert("RGB")
    w, h   = pil.size
    cx, cy = w//2, h//2
    ox, oy = int(w*0.10), int(h*0.10)
    return [
        rsz(pil),
        rsz(pil.crop((0, 0, cx+ox, cy+oy))),
        rsz(pil.crop((cx-ox, 0, w, cy+oy))),
        rsz(pil.crop((0, cy-oy, cx+ox, h))),
        rsz(pil.crop((cx-ox, cy-oy, w, h))),
    ]

def build_prompt(question, options):
    opts = "\n".join(f"{i+1}. {o}" for i,o in enumerate(options))
    return (
        "You are a geographic expert analyzing a map image.\n\n"
        f"Question: {question}\n\nOptions:\n{opts}\n\n"
        "End your response with exactly:  ANSWER: <digit>\n"
        "where <digit> is 1, 2, 3, or 4. If unsure write: ANSWER: 5\n"
    )

def extract_answer(text):
    m = re.search(r"ANSWER\s*[:=]\s*([1-5])", str(text), re.IGNORECASE)
    if m: return int(m.group(1))
    digits = re.findall(r"[1-5]", str(text)) 
    return int(digits[-1]) if digits else 5

def internvl_answer(model, tokenizer, tiles, question, options):
    from collections import Counter
    prompt, gen_cfg = build_prompt(question, options), dict(max_new_tokens=64, do_sample=False)
    answers = []
    for i, tile_pil in enumerate(tiles):
        try:
            pv = pil_to_pixel_values(tile_pil, model)
            response = model.chat(tokenizer, pv, prompt, gen_cfg)
            answers.append(extract_answer(response))
        except: pass

    counts = Counter(a for a in answers if a != 5)
    return counts.most_common(1)[0][0] if counts else 5

def load_internvl(model_dir):
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    import json
    cpath = os.path.join(model_dir, "config.json")
    if os.path.exists(cpath):
        with open(cpath, "r") as f: cfg = json.load(f)
        cfg["auto_map"] = {
            "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
            "AutoModel": "modeling_internvl_chat.InternVLChatModel",
            "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
        }
        with open(cpath, "w") as f: json.dump(cfg, f, indent=2)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, local_files_only=True)
    if vram_gb < 20:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModel.from_pretrained(model_dir, quantization_config=bnb, low_cpu_mem_usage=True, 
                                          trust_remote_code=True, local_files_only=True).eval()
    else:
        model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, 
                                          trust_remote_code=True, local_files_only=True).eval().cuda()
    return model, tokenizer

def main():
    args = parse_args()
    test_dir, script_dir = Path(args.test_dir).resolve(), Path(__file__).resolve().parent
    output_csv, cache = Path("submission.csv"), str(script_dir / "stitched_map.png")
    
    patches_path = test_dir / "patches"
    if not patches_path.exists(): patches_path = test_dir

    try: stitched = stitch(str(patches_path), cache)
    except:
        fb = str(patches_path / "patch_0.png")
        stitched = cv2.imread(fb); cv2.imwrite(cache, stitched)

    pil_full = bgr_to_pil(stitched)
    tiles = make_tiles(pil_full, max_dim=1344)
    vqa_model, vqa_proc = load_internvl(str(script_dir / "models" / "InternVL2-8B"))

    test_csv = test_dir / "test.csv"
    test_df = pd.read_csv(test_csv)
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        qid = str(row["id"])
        try:
            opts = [str(row[f"option_{i}"]) for i in range(1,5)]
            opts = [o for o in opts if o not in ("nan","None","")]
            ans  = internvl_answer(vqa_model, vqa_proc, tiles, str(row["question"]), opts)
        except: ans = 5
        results.append({"id":qid, "question_num":qid, "option":ans})

    pd.DataFrame(results, columns=["id","question_num","option"]).to_csv(str(output_csv), index=False)

if __name__ == "__main__":
    main()
