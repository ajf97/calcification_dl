import sys

import numpy as np
import pytest
from torchvision.transforms import Compose

# Insert the path of modules folder
sys.path.insert(
    0, "C:\\Users\\ajf97\\root\\software\\repositories\\calcification_dl\\src"
)

from utils import segmentation_map


def test_segmentation_map():
    image = np.random.rand(100, 100)
    prediction = segmentation_map(image)
    assert np.all(prediction >= 0) and np.all(prediction <= 255)
