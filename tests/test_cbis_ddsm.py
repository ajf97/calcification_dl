import os
import sys

import numpy as np
import pytest

# Insert the path of modules folder
sys.path.insert(
    0, "C:\\Users\\ajf97\\root\\software\\repositories\\calcification_dl\\src"
)

from data.cbis_ddsm import CBISDataset


def test_training_data_len():
    training_data = CBISDataset(
        "datasets\\CBIS-DDSM\\train\\images",
        "datasets\\CBIS-DDSM\\train\\masks",
    )
    assert len(training_data) == 30


def test_test_data_len():
    test_data = CBISDataset(
        "datasets\\CBIS-DDSM\\test\\images", "..\\datasets\\CBIS-DDSM\\test\\masks"
    )
    assert len(test_data) == 7


def test_val_data_len():
    test_data = CBISDataset(
        "datasets\\CBIS-DDSM\\validation\\images",
        "datasets\\CBIS-DDSM\\validation\\masks",
    )
    assert len(test_data) == 4


def test_get_item():
    training_data = CBISDataset(
        "datasets\\CBIS-DDSM\\train\\images",
        "datasets\\CBIS-DDSM\\train\\masks",
    )
    image, label = training_data[0]

    assert type(image) == np.ndarray
    assert len(image) > 0
    assert type(label) == np.ndarray
    assert len(label) > 0
