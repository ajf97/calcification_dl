import os
import sys

import numpy as np
import pytest

# Insert the path of modules folder
sys.path.insert(
    0, "C:\\Users\\ajf97\\root\\software\\repositories\\calcification_dl\\src"
)

from data.kios import KIOSDataset


def test_training_data_len():
    training_data = KIOSDataset(
        "datasets\\KIOS\\train\\images",
        "datasets\\KIOS\\train\\masks",
    )
    assert len(training_data) == 79


def test_test_data_len():
    test_data = KIOSDataset(
        "datasets\\KIOS\\test\\images", "datasets\\KIOS\\test\\masks"
    )
    assert len(test_data) == 12


def test_val_data_len():
    test_data = KIOSDataset(
        "datasets\\KIOS\\validation\\images",
        "datasets\\KIOS\\validation\\masks",
    )
    assert len(test_data) == 8


def test_get_item():
    training_data = KIOSDataset(
        "datasets\\KIOS\\train\\images",
        "datasets\\KIOS\\train\\masks",
    )
    image, label = training_data[0]

    assert type(image) == np.ndarray
    assert len(image) > 0
    assert type(label) == np.ndarray
    assert len(label) > 0
