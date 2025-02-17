import pickle
from itertools import product

import cv2 as cv
import numpy as np

from app.enums.colors import Color


class KNNClassifier:
    centers = 1 / 6, 3 / 6, 5 / 6
    patch_size = 8

    def __init__(self, model_path: str) -> None:
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def _get_color_index(self, image: np.ndarray, center: tuple) -> Color:
        image_height, image_width, _ = image.shape
        center_y, center_x = center

        start_y = int(image_height * center_y - self.patch_size / 2)
        end_y = start_y + self.patch_size

        start_x = int(image_width * center_x - self.patch_size / 2)
        end_x = start_x + self.patch_size

        image_patch = image[start_y:end_y, start_x:end_x]
        patch_color = np.mean(image_patch, axis=(0, 1)).reshape(1, -1)

        return self.model.predict(patch_color)[0]

    def get_colors(self, image: np.ndarray) -> list[Color]:
        if image.size == 0:
            return []

        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

        return [Color(self._get_color_index(image, (center_y, center_x)))
                for center_y, center_x in product(self.centers, repeat=2)]
