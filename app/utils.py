from itertools import product

import numpy as np

from app.enums.colors import Color


def get_virtual_cube(colors: list[Color], cube_size: int) -> np.ndarray:
    centers = 1 / 6, 3 / 6, 5 / 6

    patch_height = int(cube_size / 3  - 2)
    patch_width = int(cube_size / 3 - 2)

    virtual_cube = np.zeros((cube_size, cube_size, 3)).astype(np.uint8)

    for color, (center_y, center_x) in zip(colors, product(centers, repeat=2)):
        start_y = int(center_y * cube_size - patch_height / 2)
        end_y = start_y + patch_height

        start_x = int(center_x * cube_size - patch_width / 2)
        end_x = start_x + patch_width

        color_patch = np.tile(color.bgr, (patch_height, patch_width, 1))
        virtual_cube[start_y:end_y, start_x:end_x] = color_patch

    return virtual_cube
