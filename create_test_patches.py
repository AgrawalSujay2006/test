import argparse, os, random
import cv2, numpy as np
from pathlib import Path
 
def create_patches(input_path, output_dir, rows=5, cols=5, rotate_prob=0.6):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    h, w   = img.shape[:2]
    ph, pw = h // rows, w // cols
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Image: {w}x{h} | Grid: {rows}x{cols} | Patch: {pw}x{ph}")
 
    ROTS = [(0, None),
            (90,  cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]
 
    patches = [{"patch": img[r*ph:(r+1)*ph, c*pw:(c+1)*pw], "row": r, "col": c}
               for r in range(rows) for c in range(cols)]
    anchor, rest = patches[0], patches[1:]
    random.shuffle(rest)
 
    cv2.imwrite(os.path.join(output_dir, "patch_0.png"), anchor["patch"])
    print("patch_0.png -> top-left anchor | 0 deg [ANCHOR]")
 
    for i, p in enumerate(rest, 1):
        img_p = p["patch"]; deg = 0
        if random.random() < rotate_prob:
            deg, code = random.choice(ROTS[1:])
            img_p = cv2.rotate(img_p, code)
        cv2.imwrite(os.path.join(output_dir, f"patch_{i}.png"), img_p)
        print(f"patch_{i}.png -> (r={p[\"row\"]}, c={p[\"col\"]}) | {deg} deg")
    print(f"\n✓ {len(patches)} patches saved to {output_dir}/")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output_dir",  default="./test_patches")
    ap.add_argument("--rows",        type=int,   default=5)
    ap.add_argument("--cols",        type=int,   default=5)
    ap.add_argument("--rotate_prob", type=float, default=0.6)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    create_patches(args.input, args.output_dir,
                   args.rows, args.cols, args.rotate_prob)
