

### Create and activate a Python environment

```bash
python3 -m venv gclip_env
source gclip_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Conda**
```bash
conda create -n gclip_env python=3.10 -y
conda activate gclip_env
pip install -r requirements.txt
```

## Datasets

Create the base directories:


### 1) Compositional Visual Genome (ComVG)

- Place in: `datasets/ComVG/`

### 2) Flickr30k

- Download images: https://shannon.cs.illinois.edu/DenotationGraph/
- Store images in: `datasets/Flickr/flickr30k_images/`
- Save test pickle as: `datasets/Flickr/flickr30k_test.pkl`
- Save entities as: `datasets/Flickr/flickr30k_entities`

### 3) SVO-Probes

- Download CSV: https://github.com/google-deepmind/svo_probes/blob/main/svo_probes.csv
- Store images in: `datasets/SVO/images/`
- Save CSV as: `datasets/SVO/svo-probes.csv`

### 4) MS-COCO

- Download images + annotations: https://cocodataset.org/#download
- Save to: `datasets/MSCOCO/`
- Instead of downloading the coco images and annotations, run the coco preprocessing, it saves the images and captions

---

## Dataset Preparation

### Flickr30k

```bash
python Preprocessing/flickr_preprocess.py
```

- Output: cleaned captions saved to `datasets/Flickr/captions/`

### SVO-Probes

```bash
python Preprocessing/SVO_preprocess.py
```

- Output: triplets saved to `datasets/SVO/`

### MS-COCO

```bash
python Preprocessing/MSCOCO_preprocess.py
```

> **Note**  
> For both **Flickr30k** and **MSCOCO**, preprocessing also includes CLIP ViT-L retrieval and retrieves **top 9 images** for each caption along with the original image (**top 10 total**).  
> Outputs: `flickr_test_top10.csv` and `coco_test_top10.csv`

---

## Parsing and Phrases
```bash
python Preprocessing/parse_relation.py
python Preprocessing/get_phrase.py
```

## Get Bounding Boxes using Grounding DINO
cd GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

```bash
bash run_gdino.sh
```
> The annotated images and the bounding box coordinates will be saved to `GroundingCLIP/GdinoOutput`

## Image Text Matching

```bash
bash run_gdino.sh
```
> For eacg dataset make sure to change the file paths accordingly.

---
