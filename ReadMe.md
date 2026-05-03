GNR Project: Geospatial Image Stitching and MCQ Answering

Team Details

    Student 1 Roll No: 23B2132  
    Student 2 Roll No: 23B0023
    Student 3 Roll No: 23B0733

Project Overview
This project provides an automated pipeline to reconstruct a large map from overlapping, shuffled, and rotated image patches. After reconstruction, it utilizes a local Vision-Language Model (VLM) to answer multiple-choice questions regarding spatial relationships and landmarks within the map.

Pipeline Architecture
    Image Stitching (stitcher.py):

        Primary Stage: SIFT feature detection and FLANN matching with RANSAC-based homography.  

        Fallback Stage: A perimeter-proof jigsaw solver that minimizes Mean Absolute Difference (MAD) across overlapping regions,           ensuring a valid map is always generated even for textureless areas.  

VQA Inference (inference.py):
    Uses InternVL2-8B as the primary engine.  

    Implements a tiling strategy to process high-resolution crops, ensuring small text labels are legible.  

    Features a dynamic Dtype Safety check to ensure compatibility with 48GB L40s hardware in bfloat16 or quantized modes.

Environment Setup
The environment is initialized via the setup.bash script provided in the submission zip.

One-time Setup (Internet Required)
bash setup.bash

This script performs the following actions:
    Clones the repository and creates the gnr_project_env conda environment with Python 3.11.  

    Installs torch (compatible with CUDA 12.6), transformers, and other required libraries.  

    Downloads the required model weights (InternVL2-8B) to the local models/ directory.  

    Patches configurations to ensure the system remains strictly offline during evaluation.

Running Inference (No Internet Required)
    The inference script reads from a test directory containing a patches/ folder and a test.csv file

      conda activate gnr_project_env
      python inference.py --test_dir <absolute_path_to_test_dir>

Output
The script generates a submission.csv in the current working directory following the required format:  
    id,question_num,option  

    Option values: 1, 2, 3, 4 (Attempted) or 5 (Unanswered/Skip).


Competition Compliance
    Top-Left Anchor: The logic strictly respects patch_0.png as the coordinate anchor for the top-left corner.  

    Hallucination Protection: Parsing logic is constrained to values 1–5; any uncertainty defaults to 5 to avoid hallucination           penalties.  

    Offline Execution: Environment variables are configured within the setup and inference scripts to prevent all network calls         during the evaluation stage.

Citations
    InternVL2: Chen, K., et al. (2024). InternVL: Scaling up Vision Foundation Models.  

    LLaVA: Liu, H., et al. (2023). Visual Instruction Tuning.  

    OpenCV: Bradski, G. (2000). The OpenCV Library.
