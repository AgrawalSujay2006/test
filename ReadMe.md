# GNR Project — Geospatial Image Stitching & MCQ Answering

## Team Members
- Roll: `YOUR_ROLL_1`
- Roll: `YOUR_ROLL_2`

## Project Overview
This project reconstructs a large map from overlapping image patches and answers
multiple-choice questions about the map using a local Vision-Language Model (VLM).

**No internet is required at inference time.**

---

## Environment Setup

### Requirements
- Linux (tested on Ubuntu 22.04)
- CUDA 12.6 (L40s GPU with 48 GB VRAM)
- Conda installed and available on PATH

### One-time setup (internet required)

```bash
bash setup.bash
```

This will:
1. Create the conda environment `gnr_project_env` with Python 3.11
2. Install all Python dependencies (PyTorch, Transformers, OpenCV, etc.)
3. Clone this repository
4. Download InternVL2-8B model weights (~16 GB)
5. Download LLaVA-1.5-7B model weights (~14 GB, fallback)

---

## Running Inference (no internet needed)

```bash
conda activate gnr_project_env
cd ~/gnr_map_project
python inference.py --test_dir <absolute_path_to_test_dir>
```

This produces `submission.csv` in the current working directory.

### Expected test directory structure:
```
<test_dir>/
├── patches/
│   ├── patch_0.png      ← always top-left anchor
│   ├── patch_1.png
│   └── ...
├── test.csv
└── sample_submission.csv
```

---

## Pipeline

### Step 1 — Image Stitching (`stitcher.py`)
- **SIFT feature matching** (primary): Detects keypoints, matches them between
  patches using FLANN, estimates homography with RANSAC, and warps patches into
  an expanding canvas. Handles rotated patches automatically.
- **Edge-similarity grid** (fallback): Places patches in a grid by minimising
  mean-absolute-difference between touching edges. Tries all 4 rotations per patch.
  Always produces output.

### Step 2 — VQA (`inference.py`)
- Loads **InternVL2-8B** (primary) or **LLaVA-1.5-7B** (fallback) — both run
  fully offline.
- Passes the stitched map image + MCQ prompt to the model.
- Extracts digit answer (1–4) from model output; defaults to 5 (unanswered) if
  no digit is found — avoiding hallucination penalty.

---

## File Structure

```
gnr_map_project/
├── inference.py          ← main entry point
├── stitcher.py           ← image stitching logic
├── setup.bash            ← environment setup (run with internet)
├── README.md             ← this file
└── models/
    ├── InternVL2-8B/     ← downloaded by setup.bash
    └── llava-1.5-7b-hf/  ← downloaded by setup.bash (fallback)
```

---

## Notes
- `patch_0.png` is always the top-left corner of the map (per competition spec).
- All other patches may be shuffled and/or rotated by 0°, 90°, 180°, or 270°.
- Submission format: `id,question_num,option` — options are 1–4, or 5 to skip.
- Output value 5 = unanswered (0 points, no penalty).
- Any value outside 1–5 = hallucinated (−1 point penalty) — our code never produces these.

## Citations
- InternVL2: Chen et al., "InternVL: Scaling up Vision Foundation Models" (2024)
- LLaVA: Liu et al., "Visual Instruction Tuning" (NeurIPS 2023)
- OpenCV: Bradski, G. (2000). The OpenCV Library.