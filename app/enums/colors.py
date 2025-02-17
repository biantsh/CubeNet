from enum import Enum


class Color(Enum):
    WHITE = 0, 'U', (255, 255, 255)
    RED = 1, 'R', (0, 0, 255)
    GREEN = 2, 'F', (0, 128, 0)
    YELLOW = 3, 'D', (0, 255, 255)
    ORANGE = 4, 'L', (0, 165, 255)
    BLUE = 5, 'B', (255, 0, 0)

    def __new__(cls, value: int, char: str, bgr: tuple) -> 'Color':
        color = object.__new__(cls)
        color._value_ = value

        color.char = char
        color.bgr = bgr

        return color
