import os
import cv2
import random
import numpy as np


# 1. IMAGE RESIZING
def resize_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return cv2.resize(image, size)


# 2. CLAHE
def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


# 3. NORMALIZATION
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image.astype(np.float32) / 255.0

    return (image - MEAN) / STD


# 4. DATA AUGMENTATION
def augment_image(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    height, width = image.shape[:2]
    angle = random.uniform(-15, 15)

    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT
    )

    brightness = random.uniform(0.8, 1.2)

    return cv2.convertScaleAbs(
        image,
        alpha=brightness,
        beta=0
    )


# 5. SAVE AUGMENTED IMAGE
def save_augmented_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    augmented = augment_image(image)

    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cv2.imwrite(output_path, augmented)

    return augmented
