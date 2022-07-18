import os

import numpy as np
import pytest
from hydra import compose, initialize

from paths import CONFIG_FOLDER_PATH
from preprocessing.target_transforms import *

with initialize(
    version_base=None,
    config_path=os.path.relpath(CONFIG_FOLDER_PATH, os.path.dirname(__file__)),
):
    cfg = compose(config_name="config")

np.random.seed(cfg.seed)


def test_sum_masks():
    mask_1: np.ndarray = np.random.randint(2, size=(100, 100))
    mask_2: np.ndarray = np.random.randint(2, size=(100, 100))
    mask_3: np.ndarray = np.random.randint(2, size=(100, 100))

    sum_mask: np.ndarray = SumMasks()(np.array([mask_1, mask_2, mask_3]))
    result: np.ndarray = np.logical_or.reduce(np.array([mask_1, mask_2, mask_3]))

    assert np.all(sum_mask == result)
    assert sum_mask.shape == result.shape


def test_single_mask():
    mask: np.ndarray = np.random.randint(2, size=(100, 100))
    sum_mask: np.ndarray = SumMasks()(np.array([mask]))
    result: np.ndarray = mask

    assert np.all(sum_mask == result)
    assert sum_mask.shape == result.shape
