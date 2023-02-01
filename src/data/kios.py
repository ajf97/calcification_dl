import os
import re

import cv2
import numpy as np
from torch.utils.data import Dataset

from preprocessing.transforms import preprocess


class KIOSDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transform=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        files_list = os.listdir(self.img_dir)
        return len(list(files_list))

    def __getitem__(self, idx):
        img_file_list = os.listdir(self.img_dir)
        img_file_name = img_file_list[idx]

        img_path = os.path.join(self.img_dir, img_file_name)
        image = cv2.imread(img_path)

        mask_file_name = img_file_name.split(".")[0]
        mask_file_name = mask_file_name + "_MASK.png"

        mask_path = os.path.join(self.mask_dir, mask_file_name)
        label = cv2.imread(mask_path, 0)

        if self.transform:
            image, label = preprocess(image, label)

        return image, label
