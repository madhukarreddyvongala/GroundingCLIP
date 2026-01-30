<h2>Environment</h2>
Create an environemt: python3 -m venv gclip_env
source gclip_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

---

<h2>Datasets</h2>
Please follow below links to download the datasets
1. Compositional Visual Genome (ComVG). Dataset is present in the datasets/ComVG
2. Flickr30k: Download the images (https://shannon.cs.illinois.edu/DenotationGraph/) and store them as datasets/Flickr/flickr30k_images and save the test pickle file as datasets/Flickr/flickr30k_test.pkl and entities as datasets/Flickr/flickr30k_entities
3. SVO-Probes: Download the images (https://github.com/google-deepmind/svo_probes/blob/main/svo_probes.csv) and store the images as datasets/SVO/images and store the csv as datasets/SVO/svo-probes.csv
4. MS-COCO: Download the images and annotations (https://cocodataset.org/#download) and save them to datasets/MSCOCO. 
---


<h2>Datasets Preparation</h2>
Please follow below steps to preprocess the data

1. Flickr30k - run Preprocessing/flickr_preprocess.py and the cleaned captions will be saved to datasets/Flickr/captions.
2. SVO Probes: run Preprocessing/SVO_preprocess.py. It saves the triplets to datasets/SVO/
3. MS COCO: run Preprocessing/MSCOCO_preprocess.py. 

For both Flickr and MSCOCO, preprocessing also includes CLIP Vit-L retrieval and retrieves top 9 images for the caption with the original image (top 10 = top 9 + 1 original image). Saved to flickr_test_top10.csv and Coco_test_top10.csv respectively.
---

<h2>Parsing and Phrases</h2>
Flickr and MSCOCO: run preprocessing/parse_relation.py and preprocessing/get_phrase.py
---

<h2>Get Bounding boxes using GroundingDINO</h2>
cd GroundingDINO
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

Run run_gdino.sh. The annotated images and the bounding box coordinates will be saved to GroundingCLIP/GdinoOutput.

---

<h2>Groudning CLIP</h2>
For Image Text Matching: Run ITM/run_glicp.sh. Make sure to change the file paths accordingly.