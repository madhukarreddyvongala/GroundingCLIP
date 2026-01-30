import os
import pickle
import shutil
import random
import csv
import json
import spacy
import openai
from tqdm import tqdm
import ast
import re

openai.api_key = "" # Add your OpenAI API key here

# base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/Flickr"))
# relations_dir = os.path.join(base_dir, "Flickr_relations")
# output_dir = os.path.join(base_dir, "Flickr_phrases")

base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/MSCOCO"))
relations_dir = os.path.join(base_dir, "Coco_relations")
output_dir = os.path.join(base_dir, "Coco_phrases")



os.makedirs(output_dir, exist_ok = True)

def extract_first_json_block(text):
    depth = 0
    for i, char in enumerate(text):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[:i+1]
    return text

def get_referring_phrase_gpt(object_name, attribute):
    prompt = (
        f"Combine the object '{object_name}' with its attribute '{attribute}' to create a natural, short referring phrase. "
        f"Do not include any explanation. Just return the phrase."
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def get_phrases():
    
    phrase_cache = {}

    for file in tqdm(os.listdir(relations_dir), desc="Processing files"):

        if not file.endswith(".json"):
            continue

        output_path = os.path.join(output_dir, file)


        if os.path.exists(output_path):
            continue

        file_path = os.path.join(relations_dir, file)
            
        with open(file_path) as f:
            data = json.load(f)


        if not isinstance(data, dict):
            try:
                data = re.sub(r'("attributes"\s*:\s*"[^"]*"),\s*"([^"]+)"',r'\1, "text": "\2"',data)
                data = json.loads(data)

            except Exception:
                try:
                    data = extract_first_json_block(data)
                    data = json.loads(data)
                except Exception:
                    continue

        object_phrases = {}
        phrases = []
        connection_phrases = []

        objects = data.get("objects", {})
        for obj, info in objects.items():
            attr = info.get("attributes")

            if isinstance(attr, list):
                attr = " and ".join(attr)
            elif attr is not None:
                attr = str(attr).strip()

            if attr and attr.lower() != "null":
                cache_key = (obj, attr)
                if cache_key in phrase_cache:
                    phrase = phrase_cache[cache_key]
                else:
                    try:
                        phrase = get_referring_phrase_gpt(obj, attr)
                    except Exception as e:
                        print(f"GPT call failed for ({obj}, {attr}): {e}")
                        phrase = f"{attr} {obj}"
                    phrase_cache[cache_key] = phrase
            else:
                phrase = obj

            object_phrases[obj] = phrase
            phrases.append(phrase)

        for conn in data.get("connections", []):
            subj = conn["subject"]
            verb = conn["verb"]
            obj = conn["object"]

            subj_phrase = object_phrases.get(subj, subj)
            obj_phrase = object_phrases.get(obj, obj)

            connection_phrases.append(f"{subj_phrase}.{verb}.{obj_phrase}")

        if len(phrases) == 1:
            joined_phrase = phrases[0]
        else:
            joined_phrase = ' '.join([phrases[0] + '.'] + [p + '.' for p in phrases[1:-1]] + [phrases[-1]])

        output = {
            "object_phrase": joined_phrase,
            "connection_phrases": connection_phrases
        }

        with open(output_path, 'w') as out_f:
            json.dump(output, out_f, indent=2)


if __name__=="__main__":
    get_phrases()