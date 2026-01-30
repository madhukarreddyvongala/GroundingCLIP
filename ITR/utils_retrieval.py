import json
from PIL import Image, ImageDraw
import numpy as np
import re

def read_image(id, image_path):
    return Image.open(image_path.format(id))

def black_outside_rectangle(image, left_top, right_bottom):
    blacked_out_image = Image.new("RGB", image.size, color="black")
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left_top, right_bottom], fill=255)
    blacked_out_image.paste(image, mask=mask)
    return blacked_out_image

def overlay_images(images):
    if not images:
        return None
    base_image = Image.new("RGB", images[0].size, (0, 0, 0))
    for image in images:
        image = image.convert("RGBA")
        image_mask = image.convert("L").point(lambda x: 255 if x > 0 else 0, mode="1")
        base_image.paste(image, (0, 0), mask=image_mask)
    return base_image


def denormalize_coordinates(box, image_width, image_height):
    cx, cy, w, h = box
    x0 = (cx - w / 2) 
    y0 = (cy - h / 2) 
    x1 = (cx + w / 2) 
    y1 = (cy + h / 2) 

    return [
        int(x0 * image_width),   
        int(y0 * image_height), 
        int(x1 * image_width),   
        int(y1 * image_height)   
    ]



def create_sub_images(row_idx, image_id, objects, bb_json, attributes, image):

    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    
    W, H = image.size

    object_prompts = {}
    
    count = 0
    for obj_name, obj_info in attributes.get("objects", {}).items():
        object_prompts[obj_name] = objects[count]
        count +=1


    object_boxes = {obj: [] for obj in object_prompts.keys()}
    for box, phrase in zip(bb_json.get("boxes", []), bb_json.get("phrases", [])):
        phrase_clean = phrase.lower().strip()
        for obj, prompt in object_prompts.items():
            if phrase_clean in prompt.lower():
                object_boxes[obj].append(box)

    
    
    object_subimages = {}
    for obj, boxes in object_boxes.items():
        subimages = []
        for box in boxes:
            left, top, right, bottom = denormalize_coordinates(box, W, H)
            subimg = black_outside_rectangle(image, (left, top), (right, bottom))
            subimages.append(subimg)
        if subimages:
            if len(subimages) > 1:
                overlaid = overlay_images(subimages)
            else:
                overlaid = subimages[0]
        else:
            overlaid = image.copy()
        object_subimages[obj] = overlaid


    relations = {}
    for idx, conn in enumerate(attributes.get("connections", [])):
        subj = conn["subject"]
        objt = conn["object"]
        verb = conn["verb"]

        if subj not in object_prompts or objt not in object_prompts:
            continue

        union_subimage = overlay_images([object_subimages.get(subj), object_subimages.get(objt)])

        verb_text = f"{verb}"
        relation_key = f"relation-{idx+1}"
        relations[relation_key] = {

            "sub":{
                "subimage":  object_subimages.get(subj),
                "text": object_prompts.get(subj, subj)
            },

            "obj":{
                "subimage": object_subimages.get(objt),
                "text": object_prompts.get(objt, objt)
            },

            "verb":{
            "subimage": union_subimage,
            "text": verb_text
            }
        }

    
    return relations




