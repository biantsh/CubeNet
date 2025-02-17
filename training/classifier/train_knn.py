import argparse
import pickle

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

COLORS = ['white', 'red', 'green', 'yellow', 'orange', 'blue']


def main(dataset_path: str, output_path: str) -> None:
    data = pd.read_csv(dataset_path)
    data.drop_duplicates(inplace=True, ignore_index=True)

    features = np.array(data.drop(['color'], axis = 1))
    target = np.array([COLORS.index(color) for color in data['color']])

    x_train, x_test, y_train, y_test = (
        train_test_split(features, target, test_size=0.1))
    model = KNeighborsClassifier(n_neighbors = 3)

    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))

    with open(output_path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)

    args = parser.parse_args()
    main(args.dataset_path, args.output_path)
