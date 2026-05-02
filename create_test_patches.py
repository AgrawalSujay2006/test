"""
create_test_patches.py
======================
Takes any large image and creates synthetic test patches similar to the
competition format:
  - patch_0.png = top-left corner, NO rotation (anchor)
  - patch_1..N  = shuffled order, random rotations (0/90/180/270)

Usage:
    python create_test_patches.py --input map.png --output_dir ./test_patches \
        --rows 5 --cols 5 --overlap 0

After running, test your stitcher:
    python stitcher.py --patches_dir ./test_patches --output result.png

Then visually compare result.png with map.png.
"""

import argparse
import os
import random
import cv2
import numpy as np
from pathlib import Path


def create_patches(input_path: str,
                   output_dir: str,
                   rows: int = 5,
                   cols: int = 5,
                   overlap: int = 0,
                   rotate_prob: float = 0.6):
    """
    Slice image into rows×cols patches and save in competition format.

    Args:
        input_path  : path to source image
        output_dir  : where to save patch_N.png files
        rows, cols  : grid dimensions
        overlap     : pixel overlap between adjacent patches (0 = no overlap)
        rotate_prob : probability of rotating a non-anchor patch
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = img.shape[:2]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute patch size (without overlap for simplicity)
    ph = h // rows
    pw = w // cols

    print(f"Image size    : {w}×{h}")
    print(f"Grid          : {rows} rows × {cols} cols")
    print(f"Patch size    : {pw}×{ph}")
    print(f"Total patches : {rows * cols}")

    ROTATIONS = [
        (0,   None),
        (90,  cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    # Collect all patches with their (row, col) origin
    patches = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * ph
            x0 = c * pw
            patch = img[y0:y0+ph, x0:x0+pw]
            is_anchor = (r == 0 and c == 0)
            patches.append({
                "patch": patch,
                "row": r,
                "col": c,
                "is_anchor": is_anchor,
            })

    # Separate anchor from rest
    anchor = patches[0]
    rest = patches[1:]

    # Shuffle the non-anchor patches
    random.shuffle(rest)

    # Save anchor as patch_0 (no rotation)
    anchor_path = os.path.join(output_dir, "patch_0.png")
    cv2.imwrite(anchor_path, anchor["patch"])
    print(f"\npatch_0.png  → top-left anchor (r=0, c=0), 0° rotation")

    # Save rest as patch_1, patch_2, ...
    for i, p in enumerate(rest, start=1):
        patch_img = p["patch"]
        rot_deg = 0
        rot_code = None

        if random.random() < rotate_prob:
            deg, code = random.choice(ROTATIONS[1:])  # skip 0°
            rot_deg  = deg
            rot_code = code
            patch_img = cv2.rotate(patch_img, rot_code)

        out_path = os.path.join(output_dir, f"patch_{i}.png")
        cv2.imwrite(out_path, patch_img)
        print(f"patch_{i}.png → origin (r={p['row']}, c={p['col']}), "
              f"rotated {rot_deg}°")

    print(f"\n✓ {len(patches)} patches saved to {output_dir}/")
    print(f"\nTo test stitching:")
    print(f"  python stitcher.py --patches_dir {output_dir} --output result.png")
    print(f"\nThen compare result.png with {input_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       required=True,
                    help="Path to source map image (any size)")
    ap.add_argument("--output_dir",  default="./test_patches",
                    help="Where to save patch_N.png files")
    ap.add_argument("--rows",        type=int, default=5)
    ap.add_argument("--cols",        type=int, default=5)
    ap.add_argument("--overlap",     type=int, default=0,
                    help="Pixel overlap (keep 0 for grid stitcher testing)")
    ap.add_argument("--rotate_prob", type=float, default=0.6,
                    help="Probability of rotating each non-anchor patch")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    create_patches(
        input_path=args.input,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        overlap=args.overlap,
        rotate_prob=args.rotate_prob,
    )