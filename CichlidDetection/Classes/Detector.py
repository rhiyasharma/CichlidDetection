import os
import time
import pandas as pd
import torch
import torchvision
import random
import csv
from torchvision.transforms import functional as F
from CichlidDetection.Classes.DataLoader import DataLoader
from CichlidDetection.Classes.FileManager import FileManager


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
        self._initiate_loader()
        self._initiate_model()
        self.dest = self.fm.local_files['weights_file']

    def _initiate_loader(self):
        """initiate train and test datasets and  dataloaders."""
        self.test_dataset = DataLoader(self._get_transform(), 'test')
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

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

    def _evaluate(self):
        """evaluate the model on the test set following an epoch of training.

        Args:
            epoch (int): epoch number, greater than or equal to 0

        """
        self.model.load_state_dict(torch.load(self.dest))
        self.model.eval()
        cpu_device = torch.device("cpu")
        results = {}
        for i, (images, targets) in enumerate(self.test_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        df = pd.DataFrame.from_dict(results, orient='index')
        df['Framefile'] = [os.path.basename(path) for path in self.test_dataset.img_files]
        df = df[['Framefile', 'boxes', 'labels', 'scores']].set_index('Framefile')
        df.to_csv(os.path.join(self.fm.local_files['predictions_dir'], 'Cichlid_detector.csv'))


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
x._evaluate()
