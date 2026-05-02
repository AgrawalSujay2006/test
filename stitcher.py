import argparse, math, os, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
 
 
def load_patches(patches_dir):
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
    yield 0,   img
    yield 90,  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    yield 180, cv2.rotate(img, cv2.ROTATE_180)
    yield 270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
 
def edge_diff(a, b, side, margin=10):
    if side == "right":
        ea = a[:, -margin:].astype(np.float32)
        eb = b[:,  :margin].astype(np.float32)
        if ea.shape[0] != eb.shape[0]: return 1e9
    else:
        ea = a[-margin:, :].astype(np.float32)
        eb = b[ :margin, :].astype(np.float32)
        if ea.shape[1] != eb.shape[1]: return 1e9
    return float(np.mean(np.abs(ea - eb)))
 
 
def sift_stitch(patches):
    detector = cv2.SIFT_create(nfeatures=1000)
    matcher  = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5}, {"checks": 50})
 
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
 
    canvas    = patches[0][1].copy()
    remaining = list(patches[1:])
    placed    = 0
    MAX_CANVAS = 25_000
 
    with tqdm(total=len(remaining), desc="  [sift] stitching") as pbar:
        while remaining:
            kp_c, des_c = kp_des(canvas)
            best = {"score": -1, "list_idx": -1, "composite": None}
 
            for li, (_, patch_img) in enumerate(remaining):
                for deg, rot_img in all_rotations(patch_img):
                    kp_p, des_p = kp_des(rot_img)
                    gm = good_matches(des_c, des_p)
                    if len(gm) < 8: continue
                    src = np.float32([kp_p[m.trainIdx].pt for m in gm]).reshape(-1,1,2)
                    dst = np.float32([kp_c[m.queryIdx].pt for m in gm]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    if H is None: continue
                    inliers = int(mask.sum())
                    if inliers < 6: continue
                    h_c, w_c = canvas.shape[:2]
                    h_p, w_p = rot_img.shape[:2]
                    corners  = np.float32([[0,0],[w_p,0],[w_p,h_p],[0,h_p]]).reshape(-1,1,2)
                    wc = cv2.perspectiveTransform(corners, H)
                    all_pts = np.concatenate([
                        np.float32([[0,0],[w_c,0],[w_c,h_c],[0,h_c]]).reshape(-1,1,2), wc])
                    x_min = int(np.floor(all_pts[:,0,0].min())) - 1
                    y_min = int(np.floor(all_pts[:,0,1].min())) - 1
                    x_max = int(np.ceil( all_pts[:,0,0].max())) + 1
                    y_max = int(np.ceil( all_pts[:,0,1].max())) + 1
                    cw, ch = x_max - x_min, y_max - y_min
                    if cw > MAX_CANVAS or ch > MAX_CANVAS: continue
                    shift  = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], dtype=np.float64)
                    warped = cv2.warpPerspective(rot_img, shift @ H, (cw, ch))
                    comp   = warped.copy()
                    comp[-y_min:-y_min+h_c, -x_min:-x_min+w_c] = canvas
                    if inliers > best["score"]:
                        best = {"score": inliers, "list_idx": li, "composite": comp}
 
            if best["composite"] is not None:
                canvas = best["composite"]
                remaining.pop(best["list_idx"])
                placed += 1
            else:
                remaining.pop(0)
            pbar.update(1)
 
    print(f"  [sift] Placed {placed}/{len(patches)-1} patches "
          f"-> {canvas.shape[1]}x{canvas.shape[0]} px")
    return canvas
 
 
def grid_stitch(patches):
    n = len(patches)
    patch_h, patch_w = patches[0][1].shape[:2]
    cols = max(1, round(math.sqrt(n * patch_w / patch_h)))
    rows = math.ceil(n / cols)
    print(f"  [grid] {rows} rows x {cols} cols  (patch {patch_w}x{patch_h} px)")
 
    remaining  = {idx: img for idx, img in patches[1:]}
    grid       = [[None]*cols for _ in range(rows)]
    grid[0][0] = patches[0][1]
 
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0: continue
            if not remaining: break
            left  = grid[r][c-1] if c > 0 else None
            above = grid[r-1][c] if r > 0 else None
            best_idx = None; best_img = None; best_score = 1e9
            for idx, img in remaining.items():
                for deg, rot_img in all_rotations(img):
                    s = 0.0; cnt = 0
                    if left  is not None: s += edge_diff(left,  rot_img, "right");  cnt += 1
                    if above is not None: s += edge_diff(above, rot_img, "bottom"); cnt += 1
                    s = s / cnt if cnt else 0.0
                    if s < best_score:
                        best_score = s; best_idx = idx; best_img = rot_img
            if best_idx is not None:
                grid[r][c] = best_img; del remaining[best_idx]
 
    canvas = np.zeros((rows*patch_h, cols*patch_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is not None:
                canvas[r*patch_h:(r+1)*patch_h,
                       c*patch_w:(c+1)*patch_w] = grid[r][c]
    print(f"  [grid] Canvas: {canvas.shape[1]}x{canvas.shape[0]} px")
    return canvas
 
 
def stitch(patches_dir, output_path="stitched_map.png", strategy="auto"):
    if os.path.exists(output_path):
        cached = cv2.imread(output_path)
        if cached is not None:
            print(f"[stitch] Loaded cached map: {output_path}")
            return cached
 
    patches = load_patches(patches_dir)
    if not patches:
        raise RuntimeError(f"No patches found in {patches_dir}")
 
    result = None
    if strategy in ("auto", "sift"):
        print("[stitch] Trying SIFT...")
        try:
            result = sift_stitch(patches)
        except Exception as e:
            print(f"  [sift] Failed: {e}"); result = None
 
    if result is None:
        print("[stitch] Using grid fallback...")
        result = grid_stitch(patches)
 
    cv2.imwrite(output_path, result)
    print(f"[stitch] Saved -> {output_path}  ({result.shape[1]}x{result.shape[0]} px)")
    return result
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="patches")
    ap.add_argument("--output",      default="stitched_map.png")
    ap.add_argument("--strategy",    default="auto",
                    choices=["auto", "sift", "grid"])
    args = ap.parse_args()
    r = stitch(args.patches_dir, args.output, args.strategy)
    print(f"Done. {r.shape[1]}x{r.shape[0]} px")
