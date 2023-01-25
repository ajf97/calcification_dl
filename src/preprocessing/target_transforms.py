import cv2
import numpy as np
from torch import argmax, from_numpy, unsqueeze


class ToTensorMask(object):
    """
    A class to convert segmentation masks to torch tensors
    """

    def __call__(self, mask):
        numpy_to_tensor = from_numpy(mask)
        return unsqueeze(numpy_to_tensor, 0)
        # return numpy_to_tensor


class ResizeMask(object):
    """
    A class to resize masks
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return cv2.resize(mask, self.size, interpolation=cv2.INTER_AREA)


class SumMasks(object):
    """
    A class to sum binary masks of calcifications belonging to same mammography.
    """

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        if len(masks.shape) == 1:
            resized_masks = []

            for mask in masks:
                resized_masks.append(
                    cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
                )

            masks_sum = np.logical_or.reduce(np.array(resized_masks)).astype(np.uint8)
        else:
            masks_sum = np.logical_or.reduce(masks).astype(np.uint8)

        return masks_sum


class PostProcess(object):
    """
    A class to post process the segmentation masks.
    """

    def __call__(self, mask):
        mask = argmax(mask, dim=1)
        mask = mask.cpu().numpy()
        mask = np.squeeze(mask)
        mask = mask * 255
        return mask
