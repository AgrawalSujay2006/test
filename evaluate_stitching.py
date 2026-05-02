"""
evaluate_stitching.py
=====================
Measures how well your stitcher reconstructed a map by comparing
the stitched result with the original image.

Metrics:
  - SSIM  (Structural Similarity)  — 1.0 = perfect
  - PSNR  (Peak Signal-to-Noise)   — higher = better, >30 dB = good
  - MSE   (Mean Squared Error)     — lower = better

Usage:
    python evaluate_stitching.py \
        --original map.png \
        --stitched result.png
"""

import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def resize_to_match(img_a, img_b):
    """Resize img_b to match img_a dimensions for comparison."""
    h, w = img_a.shape[:2]
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    return img_b


def evaluate(original_path: str, stitched_path: str):
    orig = cv2.imread(original_path)
    stit = cv2.imread(stitched_path)

    if orig is None:
        raise FileNotFoundError(f"Cannot read: {original_path}")
    if stit is None:
        raise FileNotFoundError(f"Cannot read: {stitched_path}")

    print(f"Original : {orig.shape[1]}×{orig.shape[0]} px")
    print(f"Stitched : {stit.shape[1]}×{stit.shape[0]} px")

    # Resize stitched to match original for fair comparison
    stit_r = resize_to_match(orig, stit)

    # Convert to grayscale for SSIM
    orig_gray = cv2.cvtColor(orig,   cv2.COLOR_BGR2GRAY)
    stit_gray = cv2.cvtColor(stit_r, cv2.COLOR_BGR2GRAY)

    # ── Metrics ──────────────────────────────────────────────
    mse  = float(np.mean((orig_gray.astype(np.float32) -
                           stit_gray.astype(np.float32)) ** 2))
    psnr = cv2.PSNR(orig, stit_r)
    ssim_score = ssim(orig_gray, stit_gray)

    print("\n── Stitching Quality ──────────────────")
    print(f"  SSIM : {ssim_score:.4f}  (1.0 = perfect, >0.7 = good)")
    print(f"  PSNR : {psnr:.2f} dB   (>30 dB = good)")
    print(f"  MSE  : {mse:.2f}       (lower = better)")

    if ssim_score > 0.85:
        verdict = "✓ EXCELLENT"
    elif ssim_score > 0.70:
        verdict = "✓ GOOD"
    elif ssim_score > 0.50:
        verdict = "~ ACCEPTABLE"
    else:
        verdict = "✗ POOR — check stitching logic"

    print(f"\n  Verdict: {verdict}")

    # Save side-by-side comparison
    h = min(orig.shape[0], stit_r.shape[0], 800)
    scale_o = h / orig.shape[0]
    scale_s = h / stit_r.shape[0]
    orig_disp = cv2.resize(orig,   (int(orig.shape[1]*scale_o),   h))
    stit_disp = cv2.resize(stit_r, (int(stit_r.shape[1]*scale_s), h))

    separator = np.full((h, 4, 3), 200, dtype=np.uint8)
    comparison = np.hstack([orig_disp, separator, stit_disp])

    out_path = "stitching_comparison.png"
    cv2.imwrite(out_path, comparison)
    print(f"\n  Comparison saved → {out_path}")
    print("  (Left = original, Right = stitched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="Original map image")
    ap.add_argument("--stitched", required=True, help="Stitched output image")
    args = ap.parse_args()
    evaluate(args.original, args.stitched)