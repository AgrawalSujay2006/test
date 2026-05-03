"""
stitcher.py — SIFT/FLANN Feature Stitcher + Perimeter-Proof Grid Fallback
"""
import argparse, cv2, os, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path

# ── Rotation Helpers ─────────────────────────────────────────
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
def all_rotations(img):
    yield 0, img
    for deg, code in _ROT.items():
        yield deg, cv2.rotate(img, code)

# ── Patch Loading ────────────────────────────────────────────
def load_patches(patches_dir):
    p = Path(patches_dir)
    files = sorted(p.glob("patch_*.png"), key=lambda f: int(f.stem.split("_")[1]))
    patches = {int(f.stem.split("_")[1]): cv2.imread(str(f)) for f in files}
    if not patches: raise RuntimeError(f"No patch_*.png found in {patches_dir}")
    ph, pw = next(iter(patches.values())).shape[:2]
    return patches, pw, ph

# ============================================================
# 1. PRIMARY PIPELINE: SIFT FEATURE MATCHING
# ============================================================
def warp_two_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

def stitch_sift(patches):
    print("[SIFT] Starting feature matching pipeline...")
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    canvas = patches[0]
    remaining = {k: v for k, v in patches.items() if k != 0}
    MIN_MATCH_COUNT = 15
    
    while remaining:
        kp_canvas, des_canvas = sift.detectAndCompute(canvas, None)
        if des_canvas is None: return None
            
        best_matches, best_k, best_H, best_rot_img = [], None, None, None
        
        for k, img in remaining.items():
            for deg, rot_img in all_rotations(img):
                kp_img, des_img = sift.detectAndCompute(rot_img, None)
                if des_img is None or len(kp_img) < MIN_MATCH_COUNT: continue
                    
                matches = flann.knnMatch(des_img, des_canvas, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]
                
                if len(good) > MIN_MATCH_COUNT and len(good) > len(best_matches):
                    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_canvas[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        best_matches, best_k, best_H, best_rot_img = good, k, H, rot_img
                        
        if best_k is not None:
            print(f"[SIFT] Matched patch_{best_k} ({len(best_matches)} inliers). Warping to canvas...")
            canvas = warp_two_images(canvas, best_rot_img, best_H)
            del remaining[best_k]
        else:
            print(f"[SIFT] Could not find enough matches for remaining {len(remaining)} patches.")
            print("[SIFT] FAILED. Triggering fallback.")
            return None
            
    print("[SIFT] SUCCESS: All patches stitched via feature matching.")
    return canvas

# ============================================================
# 2. FALLBACK PIPELINE: GRID EDGE-SIMILARITY
# ============================================================
def edge_diff_overlap(ref, cand, side, overlap=0.20):
    h, w = ref.shape[:2]
    ox, oy = max(1, int(w * overlap)), max(1, int(h * overlap))
    
    if side == "right":
        if ref.shape[0] != cand.shape[0]: return float('inf')
        ea, eb = ref[:, -ox:].astype(np.float32), cand[:, :ox].astype(np.float32)
    else: 
        if ref.shape[1] != cand.shape[1]: return float('inf')
        ea, eb = ref[-oy:, :].astype(np.float32), cand[:oy, :].astype(np.float32)
        
    return float(np.mean(np.abs(ea - eb)))

def evaluate_completed_grid(grid, overlap=0.20):
    """Calculates the TRUE average error across all internal edges."""
    total_err = 0.0
    edge_count = 0
    rows, cols = len(grid), len(grid[0])
    
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                total_err += edge_diff_overlap(grid[r][c], grid[r][c+1], "right", overlap)
                edge_count += 1
            if r < rows - 1:
                total_err += edge_diff_overlap(grid[r][c], grid[r+1][c], "bottom", overlap)
                edge_count += 1
                
    return total_err / max(edge_count, 1)

def attempt_grid_jigsaw(patches, rows, cols, overlap=0.20):
    n = len(patches)
    if rows * cols != n: return None, float('inf')

    grid = [[None] * cols for _ in range(rows)]
    grid[0][0] = patches[0]
    remaining = {k: v for k, v in patches.items() if k != 0}
    placed_cells = {(0, 0)}

    while remaining:
        frontier_neighbors = {}
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in placed_cells:
                    neighbors = sum(1 for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                    if (nr, nc) in placed_cells)
                    if neighbors > 0: frontier_neighbors[(r, c)] = neighbors
        
        if not frontier_neighbors: break
        
        max_n = max(frontier_neighbors.values())
        best_frontiers = [(r, c) for (r, c), count in frontier_neighbors.items() if count == max_n]

        global_best_match = None
        global_max_conf = -1.0

        for fr, fc in best_frontiers:
            matches = []
            for k, img in remaining.items():
                for deg, rot in all_rotations(img):
                    score, valid = 0.0, True
                    
                    if fr > 0 and (fr-1, fc) in placed_cells:
                        d = edge_diff_overlap(grid[fr-1][fc], rot, "bottom", overlap)
                        if d == float('inf'): valid = False
                        else: score += d
                    if valid and fr < rows-1 and (fr+1, fc) in placed_cells:
                        d = edge_diff_overlap(rot, grid[fr+1][fc], "bottom", overlap)
                        if d == float('inf'): valid = False
                        else: score += d
                    if valid and fc > 0 and (fr, fc-1) in placed_cells:
                        d = edge_diff_overlap(grid[fr][fc-1], rot, "right", overlap)
                        if d == float('inf'): valid = False
                        else: score += d
                    if valid and fc < cols-1 and (fr, fc+1) in placed_cells:
                        d = edge_diff_overlap(rot, grid[fr][fc+1], "right", overlap)
                        if d == float('inf'): valid = False
                        else: score += d
                        
                    if valid: matches.append((score / max_n, k, rot))

            if not matches: continue
            
            matches.sort(key=lambda x: x[0])
            best_err, best_k, best_rot = matches[0]
            conf = matches[1][0] - best_err if len(matches) > 1 else float('inf')

            if conf > global_max_conf:
                global_max_conf = conf
                global_best_match = (fr, fc, best_k, best_rot)

        if global_best_match is None: return None, float('inf')

        fr, fc, k, rot = global_best_match
        grid[fr][fc] = rot
        placed_cells.add((fr, fc))
        del remaining[k]

    true_error = evaluate_completed_grid(grid, overlap)
    ph, pw = patches[0].shape[:2]
    stride_y = int(ph * (1 - overlap))
    stride_x = int(pw * (1 - overlap))
    
    canvas_h = ph + (rows - 1) * stride_y
    canvas_w = pw + (cols - 1) * stride_x
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            canvas[r*stride_y : r*stride_y+ph, c*stride_x : c*stride_x+pw] = tile

    return canvas, true_error

def stitch(patches_dir, output_path="stitched_map.png", force=False):
    if not force and os.path.exists(output_path):
        img = cv2.imread(output_path)
        if img is not None: return img

    patches, pw, ph = load_patches(patches_dir)
    n = len(patches)
    
    canvas = stitch_sift(patches)
    if canvas is not None:
        cv2.imwrite(output_path, canvas)
        return canvas
        
    print("[FALLBACK] Executing Edge-Similarity Jigsaw Grid...")
    factorizations = [(i, n // i) for i in range(1, n + 1) if n % i == 0]
    valid_factorizations = []
    overlap = 0.20
    stride_y = int(ph * (1 - overlap))
    stride_x = int(pw * (1 - overlap))
    
    for rows, cols in factorizations:
        if rows == 1 or cols == 1: continue
        canvas_h = ph + (rows - 1) * stride_y
        canvas_w = pw + (cols - 1) * stride_x
        aspect_ratio = canvas_w / canvas_h
        if 0.2 <= aspect_ratio <= 5.0:
            valid_factorizations.append((rows, cols))

    best_canvas, best_score, best_dims = None, float('inf'), None
    for rows, cols in valid_factorizations:
        print(f"  [FALLBACK] Testing {rows}r x {cols}c grid...", end=" ")
        grid_canvas, avg_diff = attempt_grid_jigsaw(patches, rows, cols, overlap=overlap)
        if grid_canvas is not None:
            print(f"true_error={avg_diff:.2f}")
            if avg_diff < best_score:
                best_score, best_canvas, best_dims = avg_diff, grid_canvas, (rows, cols)
        else: print("failed")

    if best_canvas is None: best_canvas = patches[0]
    cv2.imwrite(output_path, best_canvas)
    return best_canvas

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="test_patches")
    ap.add_argument("--output", default="stitched_map.png")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stitch(args.patches_dir, args.output, force=args.force)
