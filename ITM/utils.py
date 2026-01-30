import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib.pyplot as plt
import os


def read_image(id, path_format):
    return Image.open(path_format.format(id))

def get_matching(caption_id, image_id, matched_path):
    f = open(matched_path.format(caption_id, image_id))
    result = json.load(f)
    return result    

def black_outside_rectangle(image, left_top, right_bottom):
    blacked_out_image = Image.new("RGB", image.size, color="black")
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left_top, right_bottom], fill=255)
    blacked_out_image.paste(image, mask=mask)
    return blacked_out_image

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


def create_sub_image_obj(image_id, image_path, subj, obj, bboxes):

    object_image = {}
    image = Image.open(image_path.format(image_id))

    width, height = image.size

    subject_images = []
    object_images = []
    sub_images_len = 0
    obj_images_len = 0
    subcaptions = []
    objcaptions = []

    for phrase, coordinates in bboxes.items():
        if phrase == subj:
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = denormalize_coordinates(coordinates, width, height)
            blacked_out_image = black_outside_rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y))
            subject_images.append(blacked_out_image)

        elif phrase == obj:
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = denormalize_coordinates(coordinates, width, height)
            blacked_out_image = black_outside_rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y))
            object_images.append(blacked_out_image)

    if subject_images:
        final_subject_image = overlay_images(subject_images)
        object_image[subj] = final_subject_image
        subcaptions = [subj] * len(subject_images)
        sub_images_len = len(subject_images)
    else:
        subject_images.append(image)
        subcaptions.append(None)
        final_subject_image = overlay_images(subject_images)
        object_image[subj] = final_subject_image

    if object_images:
        final_object_image = overlay_images(object_images)
        object_image[obj] = final_object_image
        objcaptions = [obj] * len(object_images)
        obj_images_len = len(object_images)
    else:
        object_images.append(image)
        objcaptions.append(None)
        final_object_image = overlay_images(object_images)
        object_image[obj] = final_object_image


    return (object_image,
            sub_images_len, obj_images_len,
            subject_images, subcaptions,
            object_images, objcaptions,
            final_subject_image, final_object_image)


def overlay_images(images):
    base_image = Image.new("RGB", images[0].size, (0, 0, 0))
    mask = Image.new("L", base_image.size, 0)
    for image in images:
        image = image.convert("RGBA")
        image_mask = image.convert("L")
        image_mask = image_mask.point(lambda x: 255 if x > 0 else 0, mode="1")
        base_image.paste(image, (0, 0), mask=image_mask)
    return base_image


def visualize_images(idx, subject_images, subcaptions,
                     object_images, objcaptions,
                     final_subject_image, final_object_image,
                     verb_image, subj, verb, obj,
                     image_id, dataset):

    num_subject_images = len(subject_images)
    num_object_images = len(object_images)

    show_final = (num_subject_images > 1) or (num_object_images > 1)

    base_rows = max(num_subject_images, num_object_images)
    grid_rows = base_rows + (1 if show_final else 0)
    grid_cols = 3  

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 5, grid_rows * 5))

    if grid_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(base_rows):
        if row < num_subject_images:
            axes[row, 0].imshow(subject_images[row])
            axes[row, 0].axis("off")
            axes[row, 0].set_title(subcaptions[row])
        else:
            axes[row, 0].axis("off")

        if row < num_object_images:
            axes[row, 1].imshow(object_images[row])
            axes[row, 1].axis("off")
            axes[row, 1].set_title(objcaptions[row])
        else:
            axes[row, 1].axis("off")

        if row == 0 and verb_image is not None:
            axes[row, 2].imshow(verb_image)
            axes[row, 2].axis("off")
            axes[row, 2].set_title(verb if verb is not None else "Verb Image")
        else:
            axes[row, 2].axis("off")

    if show_final:
        r = grid_rows - 1

        axes[r, 0].imshow(final_subject_image)
        axes[r, 0].axis("off")
        axes[r, 0].set_title("Final Subject Image")

        axes[r, 1].imshow(final_object_image)
        axes[r, 1].axis("off")
        axes[r, 1].set_title("Final Object Image")

        if verb_image is not None:
            axes[r, 2].imshow(verb_image)
            axes[r, 2].axis("off")
            axes[r, 2].set_title("Verb Image")
        else:
            axes[r, 2].axis("off")

    plt.tight_layout()

    viz_dir = f'{dataset}_Viz'
    os.makedirs(viz_dir, exist_ok=True)
    name = f"{viz_dir}/{idx}_{image_id}_{subj}_{verb}_{obj}.png"
    fig.savefig(name)
    plt.close(fig)



def create_relation_object(final_subject_image, final_object_image, sub_images_len, obj_images_len, image_id, image_path):
    if sub_images_len != 0 and obj_images_len != 0:
        verb_image = overlay_images([final_object_image, final_subject_image])
    else:
        verb_image = Image.open(image_path.format(image_id))
    return verb_image


def normalize_tensor_list(tensor_list):
    total_sum = sum(tensor.item() for tensor in tensor_list)
    normalized_list = [tensor / total_sum for tensor in tensor_list]
    return normalized_list


def evaluate(df):
    df = df[~((df['pos_score'] == 0) & (df['neg_score'] == 0))]
    print(len(df))

    accuracy_per_category = {}
    for category in df['neg_value'].unique():
        subset = df[df['neg_value'] == category]
        accuracy = (subset['pos_score'] > subset['neg_score']).mean()
        accuracy_per_category[category] = accuracy

    overall_accuracy = (df['pos_score'] > df['neg_score']).mean()

    print("Accuracy per category in neg_value:")
    for category, acc in accuracy_per_category.items():
        print(f"{category}: {acc:.4f}")

    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")