import cv2 as cv
import numpy as np

__appname__ = 'CubeNet'


class WebcamInteractor(cv.VideoCapture):
    capture_size = 960, 540
    display_size = 1440, 820

    def __init__(self) -> None:
        super().__init__(0)

        self.set(cv.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.set(cv.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])

    def get_frame(self) -> np.ndarray:
        _, frame = self.read()
        return frame

    def show_frame(self, frame: np.ndarray) -> None:
        frame = cv.resize(frame, self.display_size)
        frame = cv.flip(frame, 1)

        cv.imshow(__appname__, frame)

    def await_input(self) -> None:
        if cv.waitKey(1) & 0xFF == ord('q'):
            self.release()
