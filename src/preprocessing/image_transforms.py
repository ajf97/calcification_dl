import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class CropBorders(object):
    """
    A class to crop borders from an image.

    """

    def __init__(
        self,
        left: float = 0.01,
        right: float = 0.01,
        up: float = 0.04,
        down: float = 0.04,
    ) -> None:
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def __call__(self, image):
        n_rows, n_cols = image.shape

        left_idy = int(n_cols * self.left)
        right_idy = int(n_cols * (1 - self.right))
        up_idx = int(n_rows * self.up)
        down_idx = int(n_rows * (1 - self.down))

        return image[up_idx:down_idx, left_idy:right_idy]


class NormalizeMinMax(object):
    """
    A class to normalize an image between range

    Args:
        feature_range (tuple): The range of the output image.
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range

    def __call__(self, image):
        scaler = MinMaxScaler(feature_range=self.feature_range)
        return scaler.fit_transform(image)


class RemoveAnnotations(object):
    """
    Remove annotations from mammogram.

    Args:
        threshold (float): threshold for image binarization
        max_value (float): maximum value for pixels that are above threshold
        kernel_size (tuple): kernel size for dilation
    """

    def __init__(
        self,
        threshold: float = 0.1,
        max_value: float = 1,
        kernel_size: tuple = (23, 23),
    ):
        self.threshold = threshold
        self.max_value = max_value
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray):
        binary_image = self._binarize_image(image, self.threshold, self.max_value)
        dilated_mask = self._dilate_mask(binary_image, self.kernel_size)
        contour_mask = self._find_largest_contour(dilated_mask)
        image_without_annotations = self._apply_mask(image, contour_mask)
        return image_without_annotations

    def _binarize_image(self, image: np.ndarray, threshold: float, max_value: float):
        """Binarize an image using a threshold.

        Args:
            image (np.ndarray): input image
            threshold (float): if pixel value is greater than threshold, it is set to max_value
            max_value (float): maximum value for pixels that are above threshold

        Returns:
            _type_: binarized image
        """
        binarized_image = np.zeros(image.shape, dtype=np.uint8)
        binarized_image[image >= threshold] = max_value

        return binarized_image

    def _dilate_mask(
        self, mask: np.ndarray, kernel_size: tuple = (23, 23)
    ) -> np.ndarray:
        """Dilate a mask for a better contour detection.

        Args:
            mask (np.ndarray): input mask
            kernel_size (tuple, optional): Kernel size. Defaults to (23, 23).

        Returns:
            np.ndarray: dilated mask using morpholoical operations
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        dilated_mask = cv2.morphologyEx(open_mask, cv2.MORPH_DILATE, kernel)

        return dilated_mask

    def _sort_contours(self, contours: np.ndarray) -> list[np.ndarray]:
        """Sort contours by size

        Args:
            contours (np.ndarray): contours to sort

        Returns:
            list[np.ndarray]: sorted contours
        """
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return sorted_contours

    def _find_largest_contour(self, mask: np.ndarray) -> np.ndarray:
        """Find the largest contour in a mask.

        Args:
            mask (np.ndarray): input mask

        Raises:
            ValueError: mask must have at least one contour

        Returns:
            np.ndarray: mask with largest contour
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            sorted_contours = self._sort_contours(contours)
            largest_contour = sorted_contours[0]
            mask_black = np.zeros(mask.shape, dtype=np.uint8)

            largest_contour_mask = cv2.drawContours(
                image=mask_black,
                contours=[largest_contour],
                contourIdx=-1,
                color=1,
                thickness=-1,
            )

            return largest_contour_mask
        else:
            raise ValueError("No contours found")

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a mask to an image.

        Args:
            image (np.ndarray): input image
            mask (np.ndarray): input mask

        Returns:
            np.ndarray: image with mask applied
        """
        return cv2.bitwise_and(image, image, mask=mask)


class HistEqualization(object):
    """
    A class to equalize the histogram of an image.

    Args:
        image (np.ndarray): image to equalize
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(
            np.uint8
        )
        equalized = cv2.equalizeHist(image)
        return equalized


class Clahe(object):
    """
    A class to apply CLAHE to an image.

    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        threshold: int = 3,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.threshold = threshold

    def __call__(self, image):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(
            np.uint8
        )
        clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
        clahe_image = clahe.apply(image)

        clahe_image[clahe_image <= self.threshold] = 0
        return clahe_image


class ToSquare(object):
    """
    A class to make an image square and padding

    Args:
        image (np.ndarray): image to make square
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        output_shape = (height, width)
        position = self._left_or_right_position(image)

        if width < height:
            output_shape = (height, height)
        elif width > height:
            output_shape = (width, width)

        square_image = np.zeros(output_shape)

        if position == "left":
            square_image[:height, :width] = image
        elif position == "right":
            square_image[:height, -width:] = image

        return square_image

    def _left_or_right_position(self, image):
        if image[image.shape[0] // 2, 0] > 0:
            return "left"
        else:
            return "right"
