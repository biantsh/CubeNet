from itertools import product

import kociemba
import numpy as np

from app.enums.colors import Color


class CubeFaceDebouncer:
    threshold = 15

    def __init__(self) -> None:
        self.previous_colors = None
        self.confirmed = False
        self.count = 0

    def update(self, colors: list[Color]) -> None:
        if colors == self.previous_colors:
            self.count += 1
        else:
            self.count = 0

        self.previous_colors = colors

        if self.count >= self.threshold:
            self.confirmed = True

    def reset(self) -> None:
        self.previous_colors = None
        self.confirmed = False
        self.count = 0


class CubeInteractor:
    def __init__(self) -> None:
        self.faces = [None] * 6
        self.face_debouncer = CubeFaceDebouncer()

    def _get_state_string(self) -> str:
        state_string = ''

        for face in self.faces:
            state_string += ''.join([color.char for color in face])

        return state_string

    def _rotate_face(self, face_index: int, num_rotations: int) -> None:
        face = self.faces[face_index]

        colors = np.array(face).reshape(3, 3)
        colors = np.rot90(colors, num_rotations)

        self.faces[face_index] = list(colors.flatten())

    def _solve(self) -> str | None:
        state_string = self._get_state_string()

        try:
            return kociemba.solve(state_string)
        except ValueError:
            return None

    def register_face(self, colors: list[Color]) -> None:
        self.face_debouncer.update(colors)
        if not self.face_debouncer.confirmed:
            return

        # The face is identified by its center color
        self.faces[colors[4].value] = colors
        self.face_debouncer.reset()

        print('Registered face:', colors[4])

    def is_solvable(self) -> bool:
        return all(face for face in self.faces)

    def solve(self) -> str:
        combinations = product(range(4), repeat=6)
        original_state = self.faces.copy()

        # Try all possible face orientations until the cube is solvable
        for combination in combinations:
            self.faces = original_state.copy()

            for face_index, num_rotations in enumerate(combination):
                self._rotate_face(face_index, num_rotations)

            if solution := self._solve():
                return solution
