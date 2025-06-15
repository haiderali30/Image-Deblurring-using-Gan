import os
import cv2
import random
import numpy as np

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
    return image

def load_random_image_pairs(blur_dir, non_blur_dir, num_pairs):
    blur_files = os.listdir(blur_dir)
    non_blur_files = os.listdir(non_blur_dir)
    random.shuffle(blur_files)

    image_pairs = []
    for blur_file in blur_files[:num_pairs]:
        corresponding_non_blur_file = blur_file.replace("_blurred.png", "_gt.png")
        if corresponding_non_blur_file in non_blur_files:
            blur_img = load_and_preprocess_image(os.path.join(blur_dir, blur_file))
            non_blur_img = load_and_preprocess_image(os.path.join(non_blur_dir, corresponding_non_blur_file))
            if blur_img is not None and non_blur_img is not None:
                image_pairs.append((blur_img, non_blur_img))
    return image_pairs
