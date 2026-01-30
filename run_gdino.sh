#!/usr/bin/env bash
set -euo pipefail

# ==============================
# ======== COCO/Flickr =========
# ===============================
# DATASET="coco"   # comvg|svo|flickr|coco
# BASE_DIR="$(pwd)"
# IMAGE_DIR="${BASE_DIR}/datasets/MSCOCO/coco_test"
# CSV_FILE="${BASE_DIR}/datasets/MSCOCO/Coco_test_top10.csv"
# PHRASES_DIR="${BASE_DIR}/datasets/MSCOCO/Coco_phrases"  # only required for flickr/coco, for comvg and svo set to ""
# OUTPUT_JSON_DIR="${BASE_DIR}/GdinoOutput/${DATASET}/BoundingBoxes"
# OUTPUT_IMAGE_DIR="${BASE_DIR}/GdinoOutput/${DATASET}/BB_images"
# SAVE_ANNOTATED="--save_annotated"  # set to "" if you don’t want annotated images


# ==============================
# ======== ComVG/SVO =========
# ===============================

DATASET="svo"   # comvg|svo|flickr|coco
BASE_DIR="$(pwd)"
IMAGE_DIR="${BASE_DIR}/datasets/SVO/images"
CSV_FILE="${BASE_DIR}/datasets/SVO/svo_probes_formatted.csv"
PHRASES_DIR=""  # only required for flickr/coco, for comvg and svo set to ""
OUTPUT_JSON_DIR="${BASE_DIR}/GdinoOutput/${DATASET}/BoundingBoxes"
OUTPUT_IMAGE_DIR="${BASE_DIR}/GdinoOutput/${DATASET}/BB_images"
SAVE_ANNOTATED="--save_annotated"  # set to "" if you don’t want annotated images



python GroundingDINO/run_gdino_gclip.py \
  --dataset "$DATASET" \
  --image_dir "$IMAGE_DIR" \
  --csv_file "$CSV_FILE" \
  --output_json_dir "$OUTPUT_JSON_DIR" \
  --output_image_dir "$OUTPUT_IMAGE_DIR" \
  ${PHRASES_DIR:+--phrases_dir "$PHRASES_DIR"} \
  $SAVE_ANNOTATED

