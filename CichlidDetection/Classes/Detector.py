import os
import time
import pandas as pd
import numpy as np
import torch
import torchvision
import random
import csv
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from CichlidDetection.Classes.DataLoader import DataLoader
from CichlidDetection.Classes.FileManager import FileManager
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.autograd import Variable


def collate_fn(batch):
    """package a mini-batch of images and targets.

    Args:
        batch (list): uncollated mini-batch

    Returns:
        tuple: collated mini-batch
    """
    return tuple(zip(*batch))

class Detector:

    def __init__(self):
        # initialize detector

        self.fm = FileManager()
        # self._initiate_loader()
        self._initiate_model()
        self.dest = self.fm.local_files['weights_file']


    def get_random_images(self, num):
        """to get random images from test dataset"""
        self.detect_dataset = DataLoader(self._get_transform(), 'test')
        indices = list(range(len(self.detect_dataset)))
        np.random.shuffle(indices)
        idx = indices[:num]
        sampler = SubsetRandomSampler(idx)
        self.detect_loader = torch.utils.data.DataLoader(self.detect_dataset, sampler=sampler, batch_size=num, collate_fn=collate_fn)
        dataiter = iter(self.detect_loader)
        images, labels = dataiter.next()
        return images, labels

    def _initiate_model(self):
        """initiate the model, optimizer, and scheduler."""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3, box_detections_per_img=5)
        self.parameters = self.model.parameters()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def _get_transform(self):
        """get a composition of the appropriate data transformations.

        Args:
            train (bool): True if training the model, False if evaluating/testing the model

        Returns:
            composition of required transforms
        """
        transforms = [ToTensor()]
        return Compose(transforms)

    @torch.no_grad()
    def _evaluate(self):
        """evaluate the model on the detect set of images"""
        cpu_device = torch.device("cpu")
        self.model.load_state_dict(torch.load(self.dest, map_location=cpu_device))
        self.model.eval()
        results = {}
        for i, (images, targets) in enumerate(self.detect_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        df = pd.DataFrame.from_dict(results, orient='index')
        index_list = df.index.tolist()
        detect_framefiles = []
        for i in index_list:
            detect_framefiles.append(self.detect_dataset.img_files[i])
        df['Framefile'] = [os.path.basename(path) for path in detect_framefiles]
        df = df[['Framefile', 'boxes', 'labels', 'scores']].set_index('Framefile')
        df.to_csv('Detect_images.csv')
        df.to_csv(os.path.join(self.fm.local_files['predictions_dir'], 'detected_frames.csv'))

# helper class

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


x = Detector()
images, labels = x.get_random_images(5)
x._evaluate()
