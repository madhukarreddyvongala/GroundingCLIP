import ast
import os
import re
from PIL import Image
import pandas as pd
from tqdm import tqdm

base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/SVO"))


INPUT_CSV = os.path.join(base_dir, "svo_probes.csv")
OUTPUT_CSV = os.path.join(base_dir, "svo_probes_formatted.csv")
IMAGES_DIR = os.path.join(base_dir, "images")
POS_EXT = ".jpg"
NEG_EXT = ".jpg"

def normalize_triplet(cell):
    if pd.isna(cell):
        return ""
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x).strip().strip("'\"") for x in v)
    except Exception:
        pass
    s = s.strip().strip("[](){}")
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def truthy(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}

def determine_neg_value(row):
    if truthy(row.get("subj_neg")):
        return "subject"
    if truthy(row.get("verb_neg")):
        return "verb"
    if truthy(row.get("obj_neg")):
        return "object"
    return None

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

df = pd.read_csv(INPUT_CSV, delimiter=",", encoding="utf-8")

rows = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    pos_id = row["pos_image_id"]
    neg_id = row["neg_image_id"]
    pos_path = os.path.join(IMAGES_DIR, f"{pos_id}{POS_EXT}")
    neg_path = os.path.join(IMAGES_DIR, f"{neg_id}{NEG_EXT}")
    if not (image_ok(pos_path) and image_ok(neg_path)):
        continue
    r = row.drop(labels=[c for c in ("pos_url", "neg_url") if c in row.index])
    rows.append(r)

out = pd.DataFrame(rows)

out["neg_triplet"] = out["neg_triplet"].apply(normalize_triplet)
out["sentence_id"] = out.groupby("sentence").ngroup() + 1
out["neg_value"] = out.apply(determine_neg_value, axis=1)
out = out.drop(columns=["subj_neg", "verb_neg", "obj_neg"], errors="ignore")

if "id" not in out.columns:
    out.insert(0, "id", range(1, len(out) + 1))

cols = ["id", "sentence", "pos_triplet", "neg_triplet", "pos_image_id", "neg_image_id", "neg_value", "sentence_id"]

out = out[cols]
out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
