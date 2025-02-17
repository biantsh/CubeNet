import argparse

from app.cube import CubeInteractor
from app.models.color_classifier import KNNClassifier
from app.models.cube_detector import TFLiteDetector
from app.utils import get_virtual_cube
from app.webcam import WebcamInteractor


def main(detector_path: str, classifier_path: str) -> None:
    classifier = KNNClassifier(classifier_path)
    detector = TFLiteDetector(detector_path)

    cube = CubeInteractor()
    webcam = WebcamInteractor()

    while webcam.isOpened():
        frame = webcam.get_frame()
        detection = detector.detect(frame)

        if detection.score < 0.5:
            webcam.show_frame(frame)
            webcam.await_input()
            continue

        detection.draw(frame)
        position = detection.get_position(frame)

        top, left, bot, right = position
        image_cropped = frame[top:bot, left:right]

        colors = classifier.get_colors(image_cropped)
        cube.register_face(colors)

        if cube.is_solvable():
            print(cube.solve())
            break

        try:
            virtual_cube = get_virtual_cube(colors, (bot - top))
            frame[top:bot, left - (bot - top):left] = virtual_cube
        except ValueError:
            pass  # Out of bounds

        webcam.show_frame(frame)
        webcam.await_input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector_path', required=True)
    parser.add_argument('-c', '--classifier_path', required=True)

    args = parser.parse_args()
    main(args.detector_path, args.classifier_path)
