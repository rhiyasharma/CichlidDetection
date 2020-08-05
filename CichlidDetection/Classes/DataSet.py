from PIL import Image
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.utils import make_dir
import torch
from torch import tensor
import os
import cv2
import sys
import numpy as np
from os.path import join, basename


def read_label_file(path):
    """read box coordinates and labels from the label file and return them as a target dictionary.

    Args:
        path (str): path to label file

    Returns:
        dict: target dictionary containing the boxes tensor and labels tensor
    """
    boxes = []
    labels = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f.readlines():
                values = line.split()
                boxes.append([float(val) for val in values[:4]])
                labels.append(int(values[4]))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return {'boxes': boxes, 'labels': labels}


class DataSet(object):
    """Class to handle loading of training or testing data"""

    def __init__(self, transforms, subset):
        """initialize DataLoader

        Args:
            transforms: Composition of Pytorch transformations to apply to the data when loading
            subset (str): data subset to use, options are 'train' and 'test'
        """
        self.fm = FileManager()
        self.files_list = self.fm.local_files['{}_list'.format(subset)]
        self.img_dir = self.fm.local_files['{}_image_dir'.format(subset)]

        self.transforms = transforms

        # open either train_list.txt or test_list.txt and read the image file names
        with open(self.files_list, 'r') as f:
            self.img_files = sorted([os.path.join(self.img_dir, fname) for fname in f.read().splitlines()])
        # generate a list of matching label file names
        label_dir = self.fm.local_files['label_dir']
        self.label_files = [fname.replace('.jpg', '.txt') for fname in self.img_files]
        self.label_files = [join(label_dir, basename(path)) for path in self.label_files]

    def __getitem__(self, idx):
        """get the image and target corresponding to idx

        Args:
            idx (int): image ID number, 0 indexed

        Returns:
            tensor: img, a tensor image
            dict of tensors: target, a dictionary containing the following
                'boxes', a size [N, 4] tensor of target annotation boxes
                'labels', a size [N] tensor of target labels (one for each box)
                'image_id', a size [1] tensor containing idx
        """
        # read in the image and label corresponding to idx
        img = Image.open(self.img_files[idx]).convert("RGB")
        target = read_label_file(self.label_files[idx])
        # add idx to the target dict as 'image_id'
        target.update({'image_id': tensor([idx])})
        # apply any necessary transforms to the image and target
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)


class DetectDataSet:

    def __init__(self, transforms, img_files):
        self.img_files = sorted(img_files)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        target = {'image_id': tensor(idx)}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)


class DetectVideoDataSet:

    def __init__(self, transforms, video_file, *args):
        self.fm = FileManager()
        for i in args:
            self.pfm = i
        self.pid = self.pfm.pid
        make_dir(os.path.join(self.pfm.local_files['{}_dir'.format(self.pid)], "Frames"))
        self.img_dir = os.path.join(self.pfm.local_files['{}_dir'.format(self.pid)], "Frames")
        self.transforms = transforms
        self.img_files = []

        cap = cv2.VideoCapture(video_file)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.framerate = int(cap.get(cv2.CAP_PROP_FPS))
        self.len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frames = []

        count = 0
        for i in range(self.len):
            ret, frame = cap.read()
            if not ret:
                print('Couldnt read frame ' + str(i) in video_file + '. Ending...', file=sys.stderr)
                break
            else:
                self.frames.append(frame)
                self.img_files.append("Frame_{}.jpg".format(count))

            count += 1

        cap.release()
        self.img_files.sort()


    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        name = "Frame_{}.jpg".format(idx)
        if len(os.listdir(self.img_dir)) < self.len:
            if name not in os.listdir(self.img_dir):
                img = Image.fromarray(self.frames[idx], 'RGB')
                img.save(os.path.join(self.img_dir, name))

        img = self.frames[idx]
        target = {'image_id': tensor(idx)}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.len
