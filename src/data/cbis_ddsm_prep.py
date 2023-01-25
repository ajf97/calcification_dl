import os
import re

import cv2
import numpy as np
from torch.utils.data import Dataset


class CbisDatasetPreprocessed(Dataset):
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
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        label_path = "_".join(file_name.split("_")[:-1]) + "_MASK.png"
        label = cv2.imread(os.path.join(self.img_dir, label_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
