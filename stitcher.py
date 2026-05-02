"""
stitcher.py — Map patch stitching utilities
============================================
Three strategies tried in order:
  1. SIFT feature matching  — best for rotated/shuffled patches
  2. Edge-similarity grid   — pure fallback, always works
  3. OpenCV Stitcher        — last resort

Usage (standalone test):
    python stitcher.py --patches_dir ./patches --output stitched_map.png
"""

import argparse
import math
import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_patches(patches_dir: str):
    """
    Load all patch_N.png files sorted by index.
    patch_0 is guaranteed top-left anchor.
    Returns list of (index, np.ndarray BGR).
    """
    p = Path(patches_dir)
    files = sorted(p.glob("patch_*.png"),
                   key=lambda f: int(f.stem.split("_")[1]))
    patches = []
    for f in files:
        idx = int(f.stem.split("_")[1])
        img = cv2.imread(str(f))
        if img is not None:
            patches.append((idx, img))
    print(f"[load] {len(patches)} patches loaded from {patches_dir}")
    return patches


def all_rotations(img):
    """Yield (degrees, rotated_img) for 0°, 90°, 180°, 270°."""
    yield 0,   img
    yield 90,  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    yield 180, cv2.rotate(img, cv2.ROTATE_180)
    yield 270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def edge_diff(a, b, side, margin=10):
    """
    Mean-absolute-difference between touching edges.
    side ∈ {'right', 'bottom'}
    Lower = better match.
    """
    if side == "right":
        ea = a[:, -margin:].astype(np.float32)
        eb = b[:,  :margin].astype(np.float32)
        if ea.shape[0] != eb.shape[0]:
            return 1e9
    else:  # bottom
        ea = a[-margin:, :].astype(np.float32)
        eb = b[ :margin, :].astype(np.float32)
        if ea.shape[1] != eb.shape[1]:
            return 1e9
    return float(np.mean(np.abs(ea - eb)))


# ─────────────────────────────────────────────────────────────
# STRATEGY 1 — SIFT feature matching
# ─────────────────────────────────────────────────────────────

def sift_stitch(patches):
    """
    Greedy SIFT-based stitcher.
    patch_0 is the anchor (top-left, no rotation).
    Iteratively finds the unplaced patch with the most RANSAC inliers
    against the current canvas, warps it in, and expands the canvas.
    """
    detector = cv2.SIFT_create(nfeatures=1000)
    FLANN_INDEX_KDTREE = 1
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
        {"checks": 50}
    )

    def kp_des(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return detector.detectAndCompute(gray, None)

    def good_matches(d1, d2):
        if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
            return []
        try:
            raw = matcher.knnMatch(d1, d2, k=2)
        except cv2.error:
            return []
        return [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]

    canvas = patches[0][1].copy()
    remaining = list(patches[1:])  # list of (idx, img)
    placed = 0
    MAX_CANVAS = 25_000  # pixel safety cap

    with tqdm(total=len(remaining), desc="  [sift] stitching") as pbar:
        while remaining:
            kp_c, des_c = kp_des(canvas)
            best = {"score": -1, "list_idx": -1, "composite": None}

            for list_i, (patch_idx, patch_img) in enumerate(remaining):
                for deg, rot_img in all_rotations(patch_img):
                    kp_p, des_p = kp_des(rot_img)
                    gm = good_matches(des_c, des_p)
                    if len(gm) < 8:
                        continue

                    src_pts = np.float32(
                        [kp_p[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [kp_c[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is None:
                        continue
                    inliers = int(mask.sum())
                    if inliers < 6:
                        continue

                    # Compute new canvas size
                    h_c, w_c = canvas.shape[:2]
                    h_p, w_p = rot_img.shape[:2]
                    corners = np.float32(
                        [[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]]
                    ).reshape(-1, 1, 2)
                    warped_corners = cv2.perspectiveTransform(corners, H)
                    all_pts = np.concatenate([
                        np.float32([[0,0],[w_c,0],[w_c,h_c],[0,h_c]]).reshape(-1,1,2),
                        warped_corners
                    ])
                    x_min = int(np.floor(all_pts[:, 0, 0].min())) - 1
                    y_min = int(np.floor(all_pts[:, 0, 1].min())) - 1
                    x_max = int(np.ceil(all_pts[:, 0, 0].max())) + 1
                    y_max = int(np.ceil(all_pts[:, 0, 1].max())) + 1

                    cw = x_max - x_min
                    ch = y_max - y_min
                    if cw > MAX_CANVAS or ch > MAX_CANVAS:
                        continue

                    shift = np.array(
                        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
                    warped = cv2.warpPerspective(rot_img, shift @ H, (cw, ch))

                    # Paste canvas on top of warped patch
                    composite = warped.copy()
                    y0 = -y_min
                    x0 = -x_min
                    composite[y0:y0+h_c, x0:x0+w_c] = canvas

                    if inliers > best["score"]:
                        best = {
                            "score": inliers,
                            "list_idx": list_i,
                            "composite": composite
                        }

            if best["composite"] is not None:
                canvas = best["composite"]
                remaining.pop(best["list_idx"])
                placed += 1
            else:
                # No match found — drop first remaining patch to avoid infinite loop
                remaining.pop(0)
            pbar.update(1)

    print(f"  [sift] Placed {placed}/{len(patches)-1} patches "
          f"→ canvas {canvas.shape[1]}×{canvas.shape[0]} px")
    return canvas


# ─────────────────────────────────────────────────────────────
# STRATEGY 2 — Edge-similarity grid (always works)
# ─────────────────────────────────────────────────────────────

def grid_stitch(patches):
    """
    Arrange patches in a grid, optimising each patch's rotation via edge
    similarity. patch_0 is fixed at (0,0) — top-left anchor.
    This always produces an output, even if stitching quality is imperfect.
    """
    n = len(patches)
    patch_h, patch_w = patches[0][1].shape[:2]

    # Estimate grid dimensions — prefer square-ish layout
    cols = max(1, round(math.sqrt(n * patch_w / patch_h)))
    rows = math.ceil(n / cols)

    print(f"  [grid] Layout: {rows} rows × {cols} cols "
          f"(patch {patch_w}×{patch_h} px)")

    remaining = {idx: img for idx, img in patches[1:]}
    grid = [[None] * cols for _ in range(rows)]
    grid[0][0] = patches[0][1]

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue
            if not remaining:
                break

            left  = grid[r][c - 1] if c > 0 else None
            above = grid[r - 1][c] if r > 0 else None

            best_idx   = None
            best_img   = None
            best_score = 1e9

            for idx, img in remaining.items():
                for deg, rot_img in all_rotations(img):
                    score = 0.0
                    cnt = 0
                    if left  is not None:
                        score += edge_diff(left,  rot_img, "right")
                        cnt += 1
                    if above is not None:
                        score += edge_diff(above, rot_img, "bottom")
                        cnt += 1
                    score = score / cnt if cnt else 0.0

                    if score < best_score:
                        best_score = score
                        best_idx   = idx
                        best_img   = rot_img

            if best_idx is not None:
                grid[r][c] = best_img
                del remaining[best_idx]

    # Compose canvas
    canvas = np.zeros((rows * patch_h, cols * patch_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is not None:
                canvas[r*patch_h:(r+1)*patch_h,
                       c*patch_w:(c+1)*patch_w] = grid[r][c]

    print(f"  [grid] Canvas: {canvas.shape[1]}×{canvas.shape[0]} px")
    return canvas


# ─────────────────────────────────────────────────────────────
# STRATEGY 3 — OpenCV Stitcher (last resort)
# ─────────────────────────────────────────────────────────────

def opencv_stitch(patches):
    """
    Try OpenCV's built-in Stitcher_SCANS.
    Returns (result_bgr, True) on success, (None, False) otherwise.
    """
    imgs = [img for _, img in patches]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    try:
        status, result = stitcher.stitch(imgs)
    except cv2.error as e:
        print(f"  [opencv] cv2.error: {e}")
        return None, False

    STATUS = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "NEED_MORE_IMGS",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "HOMOGRAPHY_FAIL",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "CAMERA_PARAMS_FAIL",
    }
    msg = STATUS.get(status, f"code={status}")

    if status == cv2.Stitcher_OK:
        print(f"  [opencv] OK → {result.shape[1]}×{result.shape[0]} px")
        return result, True
    else:
        print(f"  [opencv] Failed: {msg}")
        return None, False


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def stitch(patches_dir: str,
           output_path: str = "stitched_map.png",
           strategy: str = "auto") -> np.ndarray:
    """
    Main stitching entry point.

    strategy:
        'auto'  — try sift → grid (recommended)
        'sift'  — SIFT feature matching only
        'grid'  — edge-similarity grid only (fastest)
        'opencv'— OpenCV Stitcher only
    """
    # ── Cache hit ────────────────────────────────────────────
    if os.path.exists(output_path):
        cached = cv2.imread(output_path)
        if cached is not None:
            print(f"[stitch] Loaded cached map: {output_path} "
                  f"({cached.shape[1]}×{cached.shape[0]} px)")
            return cached

    patches = load_patches(patches_dir)
    if not patches:
        raise RuntimeError(f"No patches found in {patches_dir}")

    result = None

    # ── SIFT ─────────────────────────────────────────────────
    if strategy in ("auto", "sift"):
        print("[stitch] Trying SIFT feature matching…")
        try:
            result = sift_stitch(patches)
        except Exception as e:
            print(f"  [sift] Failed: {e}")
            result = None

    # ── Grid fallback ─────────────────────────────────────────
    if result is None and strategy in ("auto", "grid"):
        print("[stitch] Using edge-similarity grid…")
        result = grid_stitch(patches)

    # ── OpenCV last resort ────────────────────────────────────
    if result is None and strategy == "opencv":
        print("[stitch] Trying OpenCV Stitcher…")
        r, ok = opencv_stitch(patches)
        if ok:
            result = r

    if result is None:
        # Absolute fallback — grid always works
        print("[stitch] Absolute fallback: grid…")
        result = grid_stitch(patches)

    cv2.imwrite(output_path, result)
    print(f"[stitch] Saved → {output_path} ({result.shape[1]}×{result.shape[0]} px)")
    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="patches")
    ap.add_argument("--output",      default="stitched_map.png")
    ap.add_argument("--strategy",    default="auto",
                    choices=["auto", "sift", "grid", "opencv"])
    args = ap.parse_args()

    result = stitch(args.patches_dir, args.output, args.strategy)
    print(f"Done. Final map: {result.shape[1]}×{result.shape[0]} px")