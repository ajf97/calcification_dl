import sys

import cv2
import numpy as np
import pytest
from torchvision.transforms import Compose

# Insert the path of modules folder
sys.path.insert(
    0, "C:\\Users\\ajf97\\root\\software\\repositories\\calcification_dl\\src"
)

from preprocessing.transforms import get_act_width, normalize


def test_normalize():
    image = np.random.rand(100, 100)
    image_normalized = normalize(image)
    assert np.all(image_normalized >= 0) and np.all(image_normalized <= 1)


def test_get_act_width():
    image = cv2.imread(
        "datasets\\CBIS-DDSM\\train\\images\\Calc-Training_P_00505_LEFT_CC_FULL.png"
    )
    width = get_act_width(image)
    assert width < image.shape[1]
