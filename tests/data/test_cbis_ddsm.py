import os
from distutils.command.config import config

import numpy as np
import pytest
from hydra import compose, initialize

from data.cbis_ddsm import CbisDataset
from paths import CONFIG_FOLDER_PATH

# Initialize config
with initialize(
    version_base=None,
    config_path=os.path.relpath(CONFIG_FOLDER_PATH, os.path.dirname(__file__)),
):
    cfg = compose(config_name="config")


def test_training_data_len():
    training_data = CbisDataset(cfg.dataset.calc_train_path)
    assert len(training_data) == 1227


def test_test_data_len():
    test_data = CbisDataset(cfg.dataset.calc_test_path)
    assert len(test_data) == 284


def test_get_item():
    training_data = CbisDataset(cfg.dataset.calc_train_path)
    image, label = training_data[0]

    assert type(image) == np.ndarray
    assert len(image) > 0
    assert type(label) == np.ndarray
    assert len(label) > 0
