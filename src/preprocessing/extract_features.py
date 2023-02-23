# %%
import cv2
import imutils
import numpy as np

from patchify import patchify


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])


def extract_features(image, feature):
    return feature(image)


def get_image_feature(patch, window_size):
    windows = sliding_window(patch, 1, window_size)

    features = []
    for w in windows:
        feature = extract_features(w[2], np.mean)
        features.append(feature)

    image_feature = np.zeros((patch.shape[0], patch.shape[1]))
    features = np.array(features).reshape((patch.shape[0], patch.shape[1]))
    for i in range(image_feature.shape[0]):
        image_feature[i] = features[i]

    return image_feature


def extract_image_features(patches, window_size):
    images_features = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            img_f = get_image_feature(patch, window_size)
            images_features.append(img_f)

    return np.array(images_features)
