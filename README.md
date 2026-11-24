
# MediMind – Chest X-ray Preprocessing

## Project Overview

MediMind is a medical image processing project focused on chest X-ray images.

This repository contains the preprocessing pipeline developed to prepare chest X-ray images for further analysis and machine learning tasks.

The dataset is divided into two categories:
- NORMAL
- PNEUMONIA

## Dataset Information

Dataset: Chest X-Ray Images (Pneumonia)

Source: Kaggle – Paul Mooney

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Original Dataset Distribution

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1341 | 3875 | 5216 |
| Test | 234 | 390 | 624 |
| Validation | 8 | 8 | 16 |
| Total | 1583 | 4273 | 5856 |

## Preprocessing Pipeline

### 1. Image Resizing
- All images are resized to 224 × 224 pixels.
- Pillow and OpenCV are used for image processing.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Applied to enhance local image contrast.
- Helps improve the visibility of image details.
- CLAHE parameters:
  - Clip Limit: 2.0
  - Tile Grid Size: 8 × 8

### 3. Image Normalization
- Pixel values are scaled from 0–255 to 0–1.
- ImageNet mean and standard deviation are applied.

Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]

- Normalized images are represented as float32 arrays.

### 4. Data Augmentation
Applied only to training images.

Techniques:
- Random horizontal flipping
- Random rotation between -15° and +15°
- Random brightness adjustment

Brightness range: 0.8–1.2

## Folder Structure

```text
MediMind/
│
├── archive (3)/
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
│
├── chest_xray_resized/
├── chest_xray_clahe/
├── chest_xray_augmented/
│
├── notebooks/
│   ├── 01_dataset_verification.ipynb
│   ├── 02_image_resizing.ipynb
│   ├── 03_clahe_preprocessing.ipynb
│   ├── 04_normalization.ipynb
│   └── 05_data_augmentation.ipynb
│
├── src/
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
```

## Installation

Make sure Python is installed.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Download or clone the MediMind repository.
2. Install the required dependencies.
3. Ensure the original dataset is available in the project directory.
4. Open the notebooks in the `notebooks` folder.
5. Execute the notebooks in numerical order.

## Reusable Preprocessing Functions

The `src/preprocessing.py` file contains reusable functions for:

- resize_image()
- apply_clahe()
- normalize_image()
- augment_image()
- save_augmented_image()

These functions can be imported into other Python scripts and notebooks.

## Important Notes

- Original images are preserved separately.
- Test and validation images are not augmented.
- Normalization returns numerical arrays and does not save separate normalized image files.
- Augmentation creates one transformed copy per training image in the current pipeline.
- The preprocessing code uses relative project paths when integrated into the final project.

## Disclaimer

MediMind is an academic project. Its outputs are intended for research and educational purposes and should not be considered a substitute for professional medical diagnosis.
