import os

import numpy as np
import pytest
from hydra import compose, initialize
from torchvision.transforms import Compose

from paths import CONFIG_FOLDER_PATH
from preprocessing.first_experiments.image_transforms import *

with initialize(
    version_base=None,
    config_path=os.path.relpath(CONFIG_FOLDER_PATH, os.path.dirname(__file__)),
):
    cfg = compose(config_name="config")

np.random.seed(cfg.seed)


def test_normalize_min_max():
    image: np.ndarray = np.random.rand(100, 100)
    normalized_image: np.ndarray = NormalizeMinMax()(image).flatten()
    assert np.all(normalized_image >= 0) and np.all(normalized_image <= 1.001)


def test_to_square():
    image: np.ndarray = np.random.rand(100, 150)
    square_image: np.ndarray = ToSquare()(image)
    assert square_image.shape[0] == square_image.shape[1]


def test_pipeline():
    image: np.ndarray = np.random.randint(256, size=(100, 150))
    pipeline: Compose = Compose(
        [CropBorders(), RemoveAnnotations(), Clahe(), ToSquare(), NormalizeMinMax()]
    )

    try:
        pipeline(image)
    except Exception as e:
        pytest.fail(f"Pipeline failed with error: {e}")
