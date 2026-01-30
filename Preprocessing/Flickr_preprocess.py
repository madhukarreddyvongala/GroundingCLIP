import os
import re
from tqdm import tqdm
import pickle
import csv
import shutil


base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/Flickr"))



SENTENCES_DIR = os.path.join(base_dir, "flickr30k_entities/Sentences")
CAPTIONS_DIR = os.path.join(base_dir, "captions")
IMAGE_DIR = os.path.join(base_dir, "flickr30k-images") 


# ==========================================
# ======  Pre Process Flickr Captions ======
#===========================================                 

entity_pattern = re.compile(r"\[/EN#\d+/\w+ (.*?)\]")

os.makedirs(CAPTIONS_DIR, exist_ok=True)

txt_files = [f for f in os.listdir(SENTENCES_DIR) if f.endswith(".txt")]

for filename in tqdm(txt_files, desc="Processing captions", unit="file"):
    input_path = os.path.join(SENTENCES_DIR, filename)
    output_path = os.path.join(CAPTIONS_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        lines = infile.readlines()
        cleaned_captions = [entity_pattern.sub(r"\1", line.strip()) for line in lines]

        outfile.write("\n".join(cleaned_captions))

print(f"\nCaptions extracted and saved in '{CAPTIONS_DIR}'")


# =================================================================
# ======  Save top 10 CLIP retrieved images for each caption ======
#================================================================== 

test_indices_path = os.path.join(base_dir,"flickr30k_test.pkl")
output_path = os.path.join(base_dir, "flickr_test_top10.csv")
TEST_IMGS_DIR = os.path.join(base_dir, "flickr_test_images")
os.makedirs(TEST_IMGS_DIR, exist_ok=True)

with open(test_indices_path, 'rb') as f:
    test_data = pickle.load(f)

index_to_image = {entry['original_index']: entry['image'] for _, entry in test_data.iterrows()}


with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'caption', 'top_10_images'])

    for _, entry in test_data.iterrows():
        image_name = entry['image']
        caption = entry['sentence']
        clip_top10_indices = [idx for idx, _ in entry['clip_baseline'][:10]]
        top10_images = [index_to_image[i] for i in clip_top10_indices if i in index_to_image]
        src_path = os.path.join(IMAGE_DIR, image_name)
        dst_path = os.path.join(TEST_IMGS_DIR, image_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Missing image: {image_name}")
            continue

        writer.writerow([os.path.join('flickr_test', image_name), caption, ';'.join(top10_images)])
        




       




