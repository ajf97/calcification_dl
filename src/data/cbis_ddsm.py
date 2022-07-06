import os
import re

import numpy as np
import pydicom
from torch.utils.data import Dataset


class CbisDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        files_list = os.listdir(self.img_dir)
        full_mammograms = filter(lambda file: "FULL" in file, files_list)
        return len(list(full_mammograms))

    def __getitem__(self, idx):
        file_list = os.listdir(self.img_dir)
        full_mammograms = list(filter(lambda file: "FULL" in file, file_list))
        file_name = full_mammograms[idx]

        img_path = os.path.join(self.img_dir, file_name)
        image = pydicom.dcmread(img_path).pixel_array.astype(np.int32)
        label = self._get_masks(file_name)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _get_masks(self, img_name: str) -> np.ndarray:
        """Get masks of full mammogram

        Args:
                img_name (str): image name of full mammogram

        Returns:
                np.ndarray: array of masks
        """
        masks = []
        file_name = img_name.split(".")[0]
        file_name = file_name.split("_")[0:-1]
        file_name = "_".join(file_name)

        # Find all masks of full mammogram
        r = re.compile(file_name + "_\d" + "_MASK.dcm")
        mask_files = list(filter(r.match, os.listdir(self.img_dir)))

        # Load masks
        for mask in mask_files:
            mask_path = os.path.join(self.img_dir, mask)
            mask = pydicom.dcmread(mask_path).pixel_array
            masks.append(mask)

        return np.array(masks)
