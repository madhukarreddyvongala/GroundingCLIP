import os
import random
import zipfile
import shutil
import pandas as pd
import requests
from tqdm import tqdm
import torch
import clip
from PIL import Image
from pycocotools.coco import COCO

base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/MSCOCO"))

IMAGE_CAPTIONS_CSV = os.path.join(base_dir, "images_and_captions.csv")
SAMPLED_CSV = os.path.join(base_dir, "COCO_test_captions.csv")
RETRIEVED_CSV = os.path.join(base_dir, "retrieved_top10.csv")
OUTPUT_CSV = os.path.join(base_dir, "Coco_test_top10.csv")

ORIGINAL_IMAGE_DIR = os.path.join(base_dir, "coco2014_val_images")
OUTPUT_IMAGE_DIR = os.path.join(base_dir, "coco_test")

N_SAMPLES = 1000
RANDOM_STATE = 42
BATCH_SIZE = 64
TOPK = 20

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMG_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

ANN_ZIP_PATH = os.path.join(base_dir, "annotations_trainval2014.zip")
IMG_ZIP_PATH = os.path.join(base_dir, "val2014.zip")
ANN_DIR = os.path.join(base_dir, "annotations")
ANN_JSON = os.path.join(ANN_DIR, "captions_val2014.json")

def image_ok(path):
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            im2.load()
        return True
    except Exception:
        return False

def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def ensure_annotations():
    os.makedirs(base_dir, exist_ok=True)
    if os.path.exists(ANN_JSON):
        return
    if not os.path.exists(ANN_ZIP_PATH):
        download_file(ANN_ZIP_URL, ANN_ZIP_PATH)
    with zipfile.ZipFile(ANN_ZIP_PATH, "r") as z:
        z.extractall(base_dir)

def ensure_images():
    os.makedirs(base_dir, exist_ok=True)
    if os.path.isdir(ORIGINAL_IMAGE_DIR) and any(n.lower().endswith(".jpg") for n in os.listdir(ORIGINAL_IMAGE_DIR)):
        return
    if not os.path.exists(IMG_ZIP_PATH):
        download_file(IMG_ZIP_URL, IMG_ZIP_PATH)
    with zipfile.ZipFile(IMG_ZIP_PATH, "r") as z:
        z.extractall(base_dir)
    extracted = os.path.join(base_dir, "val2014")
    if os.path.isdir(extracted):
        if os.path.isdir(ORIGINAL_IMAGE_DIR):
            shutil.rmtree(ORIGINAL_IMAGE_DIR, ignore_errors=True)
        shutil.move(extracted, ORIGINAL_IMAGE_DIR)

def build_images_and_captions():
    if os.path.exists(IMAGE_CAPTIONS_CSV):
        return
    ensure_annotations()
    ensure_images()
    coco = COCO(ANN_JSON)
    img_ids = coco.getImgIds()
    data = []
    for img_id in tqdm(img_ids, total=len(img_ids)):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(ORIGINAL_IMAGE_DIR, file_name)
        if not image_ok(img_path):
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        caption = random.choice(anns)["caption"]
        data.append({"image_path": file_name, "caption": caption})
    pd.DataFrame(data).to_csv(IMAGE_CAPTIONS_CSV, index=False)

def build_sampled():
    if os.path.exists(SAMPLED_CSV):
        return
    build_images_and_captions()
    df_all = pd.read_csv(IMAGE_CAPTIONS_CSV)
    if len(df_all) <= N_SAMPLES:
        df_sampled = df_all.reset_index(drop=True)
    else:
        df_sampled = df_all.sample(n=N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    df_sampled.to_csv(SAMPLED_CSV, index=False)

def encode_all_images(model, preprocess, device, all_image_files):
    feats_list = []
    valid_files = []
    for i in tqdm(range(0, len(all_image_files), BATCH_SIZE)):
        batch_files = all_image_files[i:i + BATCH_SIZE]
        imgs = []
        kept = []
        for fname in batch_files:
            p = os.path.join(ORIGINAL_IMAGE_DIR, fname)
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im).unsqueeze(0))
                kept.append(fname)
            except Exception:
                pass
        if not imgs:
            continue
        image_batch = torch.cat(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(image_batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_list.append(feats.cpu())
        valid_files.extend(kept)
    if not feats_list:
        return None, []
    return torch.cat(feats_list, dim=0).to(device), valid_files

def get_top10():
    build_sampled()
    df = pd.read_csv(SAMPLED_CSV)
    df["image_file"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
    df = df[df["image_file"].apply(lambda f: image_ok(os.path.join(ORIGINAL_IMAGE_DIR, f)))].reset_index(drop=True)
    caption_list = df["caption"].tolist()
    paired_image_files = df["image_file"].tolist()

    all_image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.lower().endswith(".jpg")])
    all_image_files = [f for f in all_image_files if image_ok(os.path.join(ORIGINAL_IMAGE_DIR, f))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    all_image_features, valid_files = encode_all_images(model, preprocess, device, all_image_files)
    if all_image_features is None:
        pd.DataFrame([], columns=["image_path", "caption", "top10images"]).to_csv(RETRIEVED_CSV, index=False)
        return

    valid_set = set(valid_files)
    caption_list2 = []
    paired_image_files2 = []
    for c, f in zip(caption_list, paired_image_files):
        if f in valid_set:
            caption_list2.append(c)
            paired_image_files2.append(f)
    caption_list = caption_list2
    paired_image_files = paired_image_files2

    text_tokens = clip.tokenize(caption_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    for i, (caption, paired_img_name) in tqdm(enumerate(zip(caption_list, paired_image_files)), total=len(caption_list)):
        query_vec = text_features[i].unsqueeze(0)
        similarity = (query_vec @ all_image_features.T).squeeze(0)
        topk_indices = similarity.topk(TOPK).indices.tolist()
        top9 = []
        for idx in topk_indices:
            candidate = valid_files[idx]
            if candidate != paired_img_name:
                top9.append(candidate)
            if len(top9) == 9:
                break
        full_top10 = [paired_img_name] + top9
        results.append({"image_path": f"{paired_img_name}", "caption": caption, "top10images": ";".join(full_top10)})

    pd.DataFrame(results).to_csv(RETRIEVED_CSV, index=False)

def separate():
    if not os.path.exists(RETRIEVED_CSV):
        get_top10()
    df = pd.read_csv(RETRIEVED_CSV)
    if df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        return


    all_images = set(df["image_path"].tolist())
    for img_list in df["top10images"].tolist():
        for img in str(img_list).split(";"):
            if img:
                all_images.add(img)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    for img in all_images:
        src = os.path.join(ORIGINAL_IMAGE_DIR, img)
        dst = os.path.join(OUTPUT_IMAGE_DIR, img)
        if image_ok(src):
            try:
                shutil.copy(src, dst)
            except Exception:
                pass

    df.to_csv(OUTPUT_CSV, index=False)

ensure_annotations()
ensure_images()
build_images_and_captions()
build_sampled()
get_top10()
separate()
