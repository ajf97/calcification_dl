import os
import re

import cv2
import numpy as np
from patchify import patchify
from torch.utils.data import Dataset

from preprocessing.transforms import preprocess


class CBISDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transform=False,
        width=300,
        extract_features=False,
        patch_size=None,
        window_size=None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.width = width
        self.extract_features = extract_features
        self.patch_size = patch_size
        self.window_size = window_size

    def __len__(self):
        files_list = os.listdir(self.img_dir)
        return len(list(files_list))

    def __getitem__(self, idx):
        img_file_list = os.listdir(self.img_dir)
        img_file_name = img_file_list[idx]

        img_path = os.path.join(self.img_dir, img_file_name)
        image = cv2.imread(img_path)

        mask_file_name = img_file_name.split(".")[0]
        mask_file_name = mask_file_name.split("_")

        mask_file_name[-1] = "MASK"
        mask_file_name = "_".join(mask_file_name) + ".png"

        mask_path = os.path.join(self.mask_dir, mask_file_name)
        label = cv2.imread(mask_path, 0)

        if self.transform:
            image, label = preprocess(image, label, self.width)

        if self.extract_features and self.patch_size and self.window_size:
            patches_image = patchify(
                image, patch_size=self.patch_size, step=self.patch_size[0]
            )

            patches_mask = patchify(
                label, patch_size=self.patch_size, step=self.patch_size[0]
            )

            image = patches_image
            label = patches_mask.reshape(
                (
                    patches_mask.shape[0] * patches_mask.shape[1],
                    patches_mask.shape[2],
                    patches_mask.shape[3],
                )
            )
        return image, label
