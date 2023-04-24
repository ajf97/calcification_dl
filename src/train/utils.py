import cv2
import numpy as np


def segmentation_map(prediction, threshold=0.5):
    seg_map = prediction.copy()
    seg_map[seg_map < threshold] = 0
    seg_map[seg_map > 0] = 1

    return seg_map * 255


def apply_mask(img, mask):
    red = np.array([255, 0, 0], dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    masked_img = np.where(mask[..., None], red, img)
    output = cv2.addWeighted(img, 0.8, masked_img, 0.6, 0)

    return output
