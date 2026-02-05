
import numpy as np
from PIL import Image

IMAGE_SIZE = (150, 150)

def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0

    if image.shape[-1] != 3:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    return image
