import cv2
import imutils
import numpy as np


def normalize(image: np.ndarray, mask=None, type=np.float32):
    image_norm = image.astype(type) / 255

    if mask is not None:
        mask_norm = mask / 255.0
        return image_norm, mask_norm
    else:
        return image_norm


def left_mamm(image: np.ndarray, mask=None):
    if image[:, :200, ...].sum() < image[:, -200:, ...].sum():
        image[:, :, ...] = image[:, ::-1, ...]

        if mask is not None:
            mask[:, :, ...] = mask[:, ::-1, ...]

    if mask is not None:
        return image, mask
    else:
        return image


def clean_mamm(image: np.ndarray) -> np.ndarray:
    background_val = 0
    image[:10, :, ...] = 0
    image[-10:, :, ...] = 0
    image[:, -10:, ...] = 0

    msk1 = (image[:, :, 0] == image[:, :, 1]) & (image[:, :, 1] == image[:, :, 2])
    image = image.mean(axis=2) * msk1
    msk = np.uint8((image > background_val) * 255)
    msk = cv2.morphologyEx(
        msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    )

    comps = cv2.connectedComponentsWithStats(msk)

    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1

    msk = (comps[1] == common_label).astype(np.uint8)

    image[:, :] = msk * image[:, :]

    return image


def cut_mamm(image: np.ndarray, mask=None):
    """Cut mamogram with an optimal width.

    We remove uneccesary background which may cause a loss of essential information

    """
    act_w = get_act_width(image)
    h = image.shape[0]
    image = image[:h, :act_w]

    if mask is not None:
        mask = mask[:h, :act_w]
        return image, mask
    else:
        return image


def get_act_width(image: np.ndarray):
    """This method obtains the size of width to cut

    Args:
        image (np.ndarray): input image

    Returns:
        w: size of width
    """
    w = image.shape[1] // 3

    while image[:, w:].max() > 0:
        w += 1

    return w


def preprocess(image: np.ndarray, mask=None, width=300):
    if mask is not None:
        image = imutils.resize(image, width)
        mask = imutils.resize(mask, width)
        image, mask = normalize(image, mask)
        image, mask = left_mamm(image, mask)
        image = clean_mamm(image)
        image, mask = cut_mamm(image, mask)
    else:
        image = imutils.resize(image, width)
        image = normalize(image)
        image = left_mamm(image)
        image = clean_mamm(image)
        image = cut_mamm(image)

    if mask is not None:
        return image, mask
    else:
        return image
