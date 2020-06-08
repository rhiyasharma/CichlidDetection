import os
import pandas as pd
from collections import defaultdict
from PIL import Image
import torch
from CichlidDetection.Classes.FileManagers import FileManager
from CichlidDetection.Utilities.utils import read_label_file


class DataLoader(object):
    def __init__(self, transforms, subset):
        self.fm = FileManager()
        self.files_list = self.fm.local_files['{}_list'.format(subset)]
        self.transforms = transforms

        with open(self.files_list, 'r') as f:
            self.img_files = sorted(f.readlines())
        self.label_files = [fname.replace('.jpg', '.txt') for fname in self.img_files]

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        target = read_label_file(self.label_files[idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)
