#!/usr/bin/env bash
set -euo pipefail

# ==============================
# ======== Flickr/COCO =========
# ===============================

# DATASET="flickr"    # "flickr or coco"
# BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# IMAGES_DIR="${BASE_DIR}/datasets/Flickr/flickr_test_images"
# CAPTIONS_PATH="${BASE_DIR}/datasets/Flickr/flickr_test_top10.csv"
# RELATIONS_PATH="${BASE_DIR}/datasets/Flickr/Flickr_relations"
# PHRASES_PATH="${BASE_DIR}/datasets/Flickr/Flickr_phrases"
# MODEL="ViT-L/14"    # ViT-L/14, RN50, ViT-B/32


DATASET="coco"    # "flickr or coco"
BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGES_DIR="${BASE_DIR}/datasets/MSCOCO/coco_test"
CAPTIONS_PATH="${BASE_DIR}/datasets/MSCOCO/Coco_test_top10.csv"
RELATIONS_PATH="${BASE_DIR}/datasets/MSCOCO/Coco_relations"
PHRASES_PATH="${BASE_DIR}/datasets/MSCOCO/Coco_phrases"
MODEL="ViT-L/14"    # ViT-L/14, RN50, ViT-B/32

python gclip_retrieval.py \
  --base_dir "$BASE_DIR" \
  --dataset "$DATASET" \
  --image_path "$IMAGES_DIR" \
  --csv_file "$CAPTIONS_PATH" \
  --relations_path "$RELATIONS_PATH" \
  --phrases_path "$PHRASES_PATH" \
  --model "$MODEL" 


