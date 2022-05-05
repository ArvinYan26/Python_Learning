import random
import numpy as np
import cv2

def random_crop(image):
    min_ratio = 0.6
    max_ratio = 1

    w, h = image.size

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image.crop((x, y, x + new_w, y + new_h))

    return image


