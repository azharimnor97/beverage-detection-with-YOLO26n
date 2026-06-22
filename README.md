# Drink Detection with YOLO

An end-to-end object detection project that trains a YOLOv8 model to detect and classify 5 beverage products in images and video. The full pipeline covers image collection, augmentation, dataset splitting, model training (Google Colab), evaluation, and video inference.

---

## Detected Classes

| ID | Class Name |
|----|------------|
| 0 | AikCheong_CoffeeMix |
| 1 | Boh_InstantTeh |
| 2 | FrencheRoast_SaltedCaramel |
| 3 | Nestum_Honey |
| 4 | Oreo_Original |

---

## Project Structure

```
object_detection_with_yolo/
│
├── script/
│   ├── image_augmentation.py         # Applies 17 augmentations per image
│   ├── image_augmentation2.py        # Lightweight version — 6 augmentations per image
│   └── split_dataset.py              # Splits augmented images into train/val/test
│
├── data/
│   ├── raw_images/
│   │   ├── sample_images/            # 32 original source images (input)
│   │   ├── augment_images/           # Output of image_augmentation.py  (544 images)
│   │   └── all_image/                # Curated augmented images fed into split_dataset.py
│   │
│   ├── annotated/                    # YOLO-format annotations from Label Studio
│   │   ├── train/
│   │   │   ├── images/               # 318 training images
│   │   │   └── labels/               # Corresponding .txt label files
│   │   ├── validation/
│   │   │   ├── images/               # 26 validation images
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/               # 40 test images
│   │       └── labels/
│   │
│   └── dataset_splited/              # Output of split_dataset.py (70/15/15 split)
│       ├── train/                    # 396 images  (~70%)
│       ├── validate/                 # 90 images   (~15%)
│       └── test/                    # 90 images   (~15%)
│
├── beverage_detection_with_YOLO26s.ipynb   # Main training & evaluation notebook (Google Colab)
├── .gitignore
└── README.md
```

---

## Pipeline Overview

```
Raw Images  →  Augmentation  →  Dataset Split  →  YOLO Training  →  Evaluation / Inference
(32 images)     (×17 or ×6)      (70/15/15)       (Google Colab)
```

### Step 1 — Image Augmentation

Two scripts are available depending on how many variants you need:

| Script | Augmentations | Output per image | Output Folder |
|--------|--------------|-----------------|---------------|
| `image_augmentation.py` | 17 | 17 variants | `augment_images/` |
| `image_augmentation2.py` | 6 | 6 variants | `augment_images2/` |

**image_augmentation.py (17 augmentations):**
Horizontal flip, Vertical flip, Brightness +60, Brightness −60, High contrast, Low contrast, Bright+contrast combined, Dark+low-contrast combined, Rotate 90°/180°/270°, Gaussian blur, Sharpen, Grayscale, Salt-and-pepper noise, Zoom-in crop, Flip+brightness combined

**image_augmentation2.py (6 augmentations):**
Horizontal flip, Vertical flip, High brightness, Low brightness, Gaussian blur, Grayscale

Both scripts preserve the original image format and follow the naming convention:
```
<original_stem>__<augmentation_type>.<ext>
# e.g. product_01__flip_h.jpeg
```

### Step 2 — Dataset Splitting

`split_dataset.py` divides all augmented images into train / validate / test partitions with **data-leakage prevention**: all augmented variants of the same source image are always assigned to the same partition, so no information leaks between sets.

Default split ratio:

| Partition | Ratio | Images |
|-----------|-------|--------|
| Train     | 70%   | 396    |
| Validate  | 15%   | 90     |
| Test      | 15%   | 90     |

### Step 3 — Model Training (Google Colab)

Training is handled in `beverage_detection_with_YOLO26s.ipynb` using the [Ultralytics](https://github.com/ultralytics/ultralytics) library.

**Model:** YOLO26s (pre-trained)  
**Hardware:** Tesla T4 GPU (15 GB VRAM)  
**Experiment Tracking:** MLflow (logs saved to Google Drive)

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Seed | 42 |

The notebook covers:
1. Mounting Google Drive and copying dataset to Colab session storage
2. Auto-generating `data.yaml` from `classes.txt`
3. Configuring MLflow experiment tracking
4. Training the model and saving checkpoints every 10 epochs
5. Running predictions on the validation and test sets
6. Evaluating and visualising mAP / precision / recall metrics
7. Running inference on a video file

---

## Model Performance

### Validation Set (26 images)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| AikCheong_CoffeeMix | 0.944 | 0.847 | 0.969 |
| Boh_InstantTeh | 1.000 | 0.868 | 0.942 |
| FrencheRoast_SaltedCaramel | 0.841 | 0.923 | 0.932 |
| Nestum_Honey | 0.946 | 1.000 | 0.994 |
| Oreo_Original | 0.942 | 1.000 | 0.995 |
| **All** | **0.935** | **0.928** | **0.966** |

**Overall mAP50-95: 0.897**

### Test Set (40 images)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| AikCheong_CoffeeMix | 0.882 | 0.857 | 0.922 |
| Boh_InstantTeh | 1.000 | 0.635 | 0.786 |
| FrencheRoast_SaltedCaramel | 0.912 | 0.850 | 0.933 |
| Nestum_Honey | 0.773 | 0.672 | 0.678 |
| Oreo_Original | 0.945 | 0.825 | 0.959 |
| **All** | **0.903** | **0.768** | **0.856** |

**Overall mAP50-95: 0.722**

---

## Setup & Usage

### Prerequisites

**Local (augmentation & splitting scripts only):**
```bash
pip install Pillow numpy
```

**Google Colab (training):**
```bash
pip install ultralytics mlflow
```

### Running the Scripts

All scripts are run from the `script/` directory.

```bash
cd script
```

**Augment source images (full — 17 augmentations):**
```bash
python image_augmentation.py
# Input:  ../data/raw_images/sample_images/
# Output: ../data/raw_images/augment_images/
```

**Augment source images (lightweight — 6 augmentations):**
```bash
python image_augmentation2.py
# Input:  ../data/raw_images/sample_images/
# Output: ../data/raw_images/augment_images2/
```

**Split dataset into train / validate / test:**
```bash
python split_dataset.py
# Input:  ../data/raw_images/all_image/
# Output: ../data/dataset_splited/
```

### Training the Model

1. Open `beverage_detection_with_YOLO26s.ipynb` in Google Colab
2. Upload or mount the dataset from Google Drive
3. Adjust hyperparameters in the configuration cells (EPOCH, SEED, batch, imgsz)
4. Run all cells top to bottom

**Run inference on images:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=<path_to_images> save=True
```

**Run inference on video:**
```bash
yolo detect predict model=<path_to_best.pt> source=<path_to_video.mp4> save=True name=predict_video
```

---

## Data Annotation

Bounding box annotations were created using **[Label Studio](https://labelstud.io/)**, an open-source data labelling platform.

Label Studio provides a web-based UI where annotators draw bounding boxes directly on images and assign class labels. It supports team collaboration (multiple annotators on the same project), built-in quality control, and can export annotations in a wide range of ML-ready formats — including YOLO format, which this project uses. The tool runs locally or on a server, and the annotated dataset is exportable as a ZIP archive containing images alongside their label files.

**Annotation workflow used in this project:**
1. Images were uploaded to a Label Studio project
2. Bounding boxes were drawn around each beverage product in every image
3. Each box was assigned one of the 5 class labels
4. The dataset was exported in YOLO format (images + `.txt` label files)
5. Exported data was placed in the `data/annotated/` directory, already split into `train/`, `validation/`, and `test/` subfolders by Label Studio

---

## Label Format

Labels follow the YOLO format — one `.txt` file per image, with one bounding box per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised (0–1) relative to the image dimensions.

---

## Ignored Files

The `.gitignore` excludes the following from version control:

- `*.jpeg` — all JPEG image files (dataset images)
- `*.zip` — archive files
- `*.avi` — video files
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv`)
- IDE configuration (`.vscode/`, `.idea/`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)