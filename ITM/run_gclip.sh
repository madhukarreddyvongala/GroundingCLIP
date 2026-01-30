#!/usr/bin/env bash
set -euo pipefail

# ==============================
# ======== ComVG/SVO =========
# ===============================

DATASET="svo"    # "comvg or svo"
BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGES_DIR="${BASE_DIR}/datasets/SVO/images"
CAPTIONS_PATH="${BASE_DIR}/datasets/SVO/svo_probes_formatted.csv"
MODEL="ViT-L/14"    # ViT-L/14, RN50, ViT-B/32
VISUALIZE=true         # This saves the background blacked out images


cmd=(python gclip.py
  --base_dir "$BASE_DIR"
  --dataset "$DATASET" 
  --image_path "$IMAGES_DIR" 
  --csv_file "$CAPTIONS_PATH" 
  --model "$MODEL" 
)

if [[ "$VISUALIZE" == true ]]; then 
  cmd+=(--viz) 
fi

"${cmd[@]}"