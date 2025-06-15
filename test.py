import cv2
import numpy as np
from generator_model import build_generator
from utils import load_and_preprocess_image

def apply_generator(generator, blur_image_path):
    image = load_and_preprocess_image(blur_image_path)
    if image is None:
        print("Failed to load image.")
        return

    output = generator(np.expand_dims(image, axis=0), training=False)[0]
    output = np.clip((output + 1) / 2, 0, 1)
    output = (output * 255).astype(np.uint8)

    return image, output
