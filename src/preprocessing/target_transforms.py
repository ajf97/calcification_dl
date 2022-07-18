import numpy as np


class SumMasks(object):
    """
    A class to sum binary masks of calcifications belonging to same mammography.
    """

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce(masks).astype(np.uint8)
