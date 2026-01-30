import torch
from utils import *
import clip
import torch
import argparse
import pandas as pd
import numpy as np
import json 
from tqdm import tqdm
import traceback, os

GPU = 0

parser = argparse.ArgumentParser(description='Image Text Matching')
parser.add_argument("--base_dir", type=str, help='path to GCLIP')
parser.add_argument("--dataset", type=str, help='dataset name')
parser.add_argument("--image_path", type=str, help='path to the images directory')
parser.add_argument("--csv_file", type=str, help='path to captions csv file')
parser.add_argument("--model", type=str, help='RN50, ViT/B-32, ViT/L-14')
parser.add_argument("--viz", action="store_true", help = "Save visualization images")
args = parser.parse_args()
IMAGE_PATH = args.image_path + "/{}.jpg" 
args.dataset = args.dataset.lower()

if args.dataset not in ["comvg", "svo"]:
    raise SystemExit("Dataset has to be either ComVG or SVO")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


data = pd.read_csv(args.csv_file)
model, preprocess = clip.load(args.model, device=device)

model.cuda(GPU).eval() 

BBOXES = f"{args.base_dir}/GdinoOutput/{args.dataset}/BoundingBoxes" + "/{}.json"

def subimage_score_embedding(image, text):
    if text:
        image = preprocess(image)
        text_input = clip.tokenize(text).cuda(GPU)
        image_input = torch.tensor(np.stack([image])).cuda(GPU)
        with torch.no_grad():
            image_embed = model.encode_image(image_input).float()
            text_embed = model.encode_text(text_input).float()
        image_features = image_embed / image_embed.norm(dim=-1, keepdim=True).float()
        text_features = text_embed /text_embed.norm(dim=-1, keepdim=True).float()
        
        score = np.array(text_features.detach().cpu().numpy()) @ np.array(image_features.detach().cpu().numpy()).T
        
        return image_embed, score[0][0]
    else:
        return None, None
    
def inference_one_pair( idx, row, image_id):
    image = preprocess(read_image(image_id, IMAGE_PATH))
    text_input = clip.tokenize(row.sentence).cuda(GPU)
    image_input = image.unsqueeze(0).cuda(GPU)
    with torch.no_grad():
        original_image_embed = model.encode_image(image_input).float()
        original_text_embed = model.encode_text(text_input).float()


    svo = row.pos_triplet.split(",")
    subj, verb, obj = svo[0], svo[1], svo[-1]

    bbox_path = BBOXES.format(image_id)
    bboxes = json.load(open(bbox_path))

    object_images, sub_images_len, obj_images_len, subject_images, subcaptions, object_imgs_list, objcaptions, final_sub_img, final_obj_img = create_sub_image_obj(
    image_id, IMAGE_PATH, subj, obj, bboxes)

    relation_image = create_relation_object(final_sub_img, final_obj_img, sub_images_len, obj_images_len, image_id, IMAGE_PATH)

    if relation_image and verb:
        object_images[verb] = relation_image

    if args.viz:
        visualize_images(
            idx,
            subject_images, subcaptions,
            object_imgs_list, objcaptions,
            final_sub_img, final_obj_img,
            relation_image,
            subj, verb, obj,
            image_id, args.dataset
        )

    ##subimages
    image_embeds = []
    image_scores = []
    for key, sub_image in object_images.items():
        image_embed, image_score = subimage_score_embedding(sub_image, key)
        if image_embed is not None and image_score is not None:
            image_embeds.append(image_embed)
            image_scores.append(image_score)
    

    respective_similarities = {}
    for score, image, key in zip(image_scores, image_embeds, object_images.keys()):
        original_image_embed += score * image
        respective_similarities[key] = score.item()

    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed /original_text_embed.norm(dim=-1, keepdim=True).float()
    
    similarity = np.array(text_features.detach().cpu().numpy()) @ np.array(image_features.detach().cpu().numpy()).T
    return similarity.item() if isinstance(similarity, np.ndarray) else similarity


def compute_one_row(idx, row):
    result_pos = inference_one_pair(idx, row, row.pos_image_id)

    result_neg = inference_one_pair(idx, row, row.neg_image_id)
    
    result = {"id" : idx, "pos_score": result_pos, "neg_score": result_neg}


    return result


if __name__ == "__main__":
    
    data["pos_score"] = 0.0
    data["neg_score"] = 0.0

    for idx, row in tqdm(data.iterrows(), total = len(data)):
        if idx == 10:
            break
        
        try:
            score = compute_one_row(idx, row)
            data.loc[idx, "pos_score"] = round(score["pos_score"], 4)
            data.loc[idx, "neg_score"] = round(score["neg_score"], 4)
        

        except Exception as e:
            print(f"Error at index {idx}: {e}", flush = True)
            traceback.format_exc()
            continue

    evaluate(data)
    
    model_name = args.model.replace('/', '_').replace('-', '_')
    filename = f"{args.dataset}_{model_name}_itm_scores.csv"
    data.to_csv(filename, index=False)

            
