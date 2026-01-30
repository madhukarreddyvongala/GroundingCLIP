import torch
from utils_retrieval import *
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
parser.add_argument("--relations_path", type=str, help='path to relations file')
parser.add_argument("--phrases_path", type=str, help='path to phrases file')
parser.add_argument("--model", type=str, help='RN50, ViT/B-32, ViT/L-14')
args = parser.parse_args()

args.dataset = args.dataset.lower()


IMAGE_PATH = args.image_path + "/{}.jpg" 
RELATIONS_PATH = args.phrases_path + "/{}.json"
ATTRIBUTES = args.relations_path + "/{}.json"
BOUNDINGBOXPATH = f"{args.base_dir}/GdinoOutput/{args.dataset}/BoundingBoxes" + "/{}_{}.json"

data = pd.read_csv(args.csv_file)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, process = clip.load(args.model, device='cpu')
model.cuda(device).eval()


def subimage_score_embedding(image, text):
    if text:
        image = process(image)
        text_input = clip.tokenize(text).cuda(device)
        image_input = torch.tensor(np.stack([image])).cuda(device)
        with torch.no_grad():
            image_embed = model.encode_image(image_input).float()
            text_embed = model.encode_text(text_input).float()
        image_features = image_embed / image_embed.norm(dim=-1, keepdim=True).float()
        text_features = text_embed /text_embed.norm(dim=-1, keepdim=True).float()
        
        score = np.array(text_features.detach().cpu().numpy()) @ np.array(image_features.detach().cpu().numpy()).T
        return image_embed, score[0][0]
    else:
        return None, None


def gclip_one_pair(row_idx, caption, image_id, relation_name):
    image = process(read_image(image_id, IMAGE_PATH))
    text_input = clip.tokenize(caption).cuda(device)
    image_input = torch.tensor(np.stack([image])).cuda(device)

    with torch.no_grad():
        original_image_embed = model.encode_image(image_input).float()
        original_text_embed = model.encode_text(text_input).float()

    try:
        relations_data = json.load(open(RELATIONS_PATH.format(relation_name)))
        objects = relations_data['object_phrase'].split(".")
        objects = list(filter(None, objects))

        attributes = json.load(open(ATTRIBUTES.format(relation_name)))
        
        bounding_box = json.load(open(BOUNDINGBOXPATH.format(row_idx, image_id)))
        
        image = Image.open(IMAGE_PATH.format(image_id))
        results = create_sub_images(row_idx, image_id, objects, bounding_box, attributes, image)

        if results:
            for key, value in results.items():
                for i in ['sub', 'obj', 'verb']:
                    subimage = value[i]['subimage']
                    text = value[i]['text']
                    
                    image_embed, score = subimage_score_embedding(subimage, text)
                    original_image_embed += score * image_embed
    except Exception as e:
        print(e)

    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed / original_text_embed.norm(dim=-1, keepdim=True).float()
    similarity = text_features.detach().cpu().numpy() @ image_features.detach().cpu().numpy().T

    return similarity[0][0]



def get_score(idx, relation_name):
    result = {}

    row = data.iloc[idx]
    candidates = row.top_10_images
    candidates = candidates.split(";")


    for candidate in candidates:
        image_name = candidate.split(".")[0]
        result[image_name] = gclip_one_pair(idx, row.caption, image_name, relation_name).item()
    
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    return result


if __name__ == "__main__":
    gclip_score = {}

    for idx, row in tqdm(data.iterrows(), total = len(data)):

        
        image_name = row.image_path.split("/")[-1].split(".")[0]
        
        try:
            gclip_score[image_name] = get_score(idx, image_name)

        except Exception as e:
            print(f"{idx} -- {e}")
            continue

    model_name = args.model.replace('/', '_').replace('-', '_')
    filename = f"{args.dataset}_{model_name}.json"


    with open(filename, "w") as f:
        json.dump(gclip_score, f, indent=2)

    top_1 = 0
    top_5 = 0
    for idx, value in gclip_score.items():
        candidates = list(value.keys())

        if args.dataset.lower() == 'flickr':
            candidates = [int(i) for i in candidates]
            if candidates[0] == int(idx):
                top_1 += 1
            if int(idx) in candidates[:5]:
                top_5 += 1

        else:
            if candidates[0] == idx:
                top_1 += 1

            if idx in candidates[:5]:
                top_5 += 1
    print("Top 1 score: {}. Top 5 score: {}".format(top_1/ len(gclip_score), top_5/ len(gclip_score)))




