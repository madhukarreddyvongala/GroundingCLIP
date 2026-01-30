import os
import csv
import json
import cv2
import argparse
import warnings
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate

warnings.filterwarnings("ignore")

MODEL_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def find_image_path(root, stem):
    stem = str(stem).strip()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(root, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None

def run_predict(model, image_path, prompt):
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    return image_source, boxes, logits, phrases

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_annotated_image(path, image_source, boxes, logits, phrases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(path, annotated_frame)

def process_itm(model, image_dir, csv_file, out_json_dir, out_img_dir, save_annotated):
    os.makedirs(out_json_dir, exist_ok=True)
    if save_annotated:
        os.makedirs(out_img_dir, exist_ok=True)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            pos_image_id = str(row.get("pos_image_id", "")).strip()
            pos_triplet = str(row.get("pos_triplet", "")).strip()
            if not pos_image_id:
                continue

            json_path = os.path.join(out_json_dir, f"{pos_image_id}.json")
            img_path = os.path.join(out_img_dir, f"{pos_image_id}.jpg")

            if save_annotated:
                if os.path.exists(json_path) and os.path.exists(img_path):
                    continue
            else:
                if os.path.exists(json_path):
                    continue

            parts = [p.strip() for p in pos_triplet.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            prompt = f"{parts[0]}. {parts[-1]}."

            image_path = find_image_path(image_dir, pos_image_id)
            if not image_path:
                continue

            
            image_source, boxes, logits, phrases = run_predict(model, image_path, prompt)
            

            result = {phrase: box.tolist() for phrase, box in zip(phrases, boxes)}
            try:
                save_json(json_path, result)
                if save_annotated:
                    save_annotated_image(img_path, image_source, boxes, logits, phrases)
            except Exception:
                continue

def process_retrieval(model, image_dir, csv_file, relations_dir, out_json_dir, out_img_dir, save_annotated):
    os.makedirs(out_json_dir, exist_ok=True)
    if save_annotated:
        os.makedirs(out_img_dir, exist_ok=True)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(reader, desc="Processing rows")):
            image_path_raw = row.get("image_path", "")
            image_file = os.path.basename(str(image_path_raw))
            image_name = os.path.splitext(image_file)[0]
            if not image_name:
                continue

            relation_path = os.path.join(relations_dir, f"{image_name}.json")
            if not os.path.exists(relation_path):
                continue

            try:
                with open(relation_path, "r") as rf:
                    rel = json.load(rf)

    
            except Exception:
                continue

            object_phrase = rel.get("object_phrase", "")
            if not object_phrase:
                continue

            lower_map = {k.lower(): k for k in row.keys()}
            top_key = None
            for cand in ("top_10_images", "top10images"):
                if cand in lower_map and row.get(lower_map[cand]):
                    top_key = lower_map[cand]
                    break

            top_str = row.get(top_key, "") if top_key else ""

            def handle_one(stem):
                img_fp = find_image_path(image_dir, stem)
                if not img_fp:
                    return

                json_path = os.path.join(out_json_dir, f"{idx}_{stem}.json")
                img_path = os.path.join(out_img_dir, f"{idx}_{stem}.jpg")

                

                if save_annotated:
                    if os.path.exists(json_path) and os.path.exists(img_path):
                        return
                else:
                    if os.path.exists(json_path):
                        return

                
                image_source, boxes, logits, phrases = run_predict(model, img_fp, object_phrase)
                
                result = {
                    "boxes": [b.tolist() for b in boxes],
                    "phrases": phrases,
                    "logits": logits.tolist() if hasattr(logits, "tolist") else logits
                }




                try:
                    save_json(json_path, result)
                    if save_annotated:
                        save_annotated_image(img_path, image_source, boxes, logits, phrases)
                except Exception:
                    return

            handle_one(image_name)

            for top_img in str(top_str).split(";"):
                top_img = top_img.strip()
                if not top_img:
                    continue
                top_stem = os.path.splitext(os.path.basename(top_img))[0]
                if top_stem:
                    handle_one(top_stem)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["comvg", "coco", "flickr", "svo"])
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--csv_file", required=True)
    ap.add_argument("--output_json_dir", required=True)
    ap.add_argument("--output_image_dir", required=True)
    ap.add_argument("--phrases_dir", default="")
    ap.add_argument("--save_annotated", action="store_true")
    args = ap.parse_args()

    model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)
    args.dataset = args.dataset.lower()

    if args.dataset == "comvg" or args.dataset == "svo":
        process_itm(model, args.image_dir, args.csv_file, args.output_json_dir, args.output_image_dir, args.save_annotated)
    else:
        if not args.phrases_dir:
            raise SystemExit("--phrases_dir is required for dataset=flickr and coco")
        process_retrieval(model, args.image_dir, args.csv_file, args.phrases_dir, args.output_json_dir, args.output_image_dir, args.save_annotated)

if __name__ == "__main__":
    main()