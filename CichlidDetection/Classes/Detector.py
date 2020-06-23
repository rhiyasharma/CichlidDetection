import os
import pandas as pd
import numpy as np
import torch
import torchvision
from CichlidDetection.Classes.DataLoader import DataLoader
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.utils import collate_fn, Compose, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler

# Run this script from the main CichlidDetection directory: python3 Detector.py
# This script selects random images from the test set and runs them through the model to produce predictions 
# Results csv file saved in scratch/CichlidDetection/training/predictions as "Detect_images.csv"



class Detector:

    def __init__(self):
        # initialize detector
        self.fm = FileManager()
        self._initiate_model()
        self.dest = self.fm.local_files['weights_file']


    def get_random_images(self, num):
        """to get random images from test dataset.
    
        Args:
        num: number of random images to select

        """
        self.detect_dataset = DataLoader(self._get_transform(), 'test')
        indices = list(range(len(self.detect_dataset)))
        np.random.shuffle(indices)
        idx = indices[:num]
        sampler = SubsetRandomSampler(idx)
        self.detect_loader = torch.utils.data.DataLoader(self.detect_dataset, sampler=sampler, batch_size=num, collate_fn=collate_fn)

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
    def evaluate(self):
        """evaluate the model on the detect set of images"""
        cpu_device = torch.device("cpu")
        # self.model.load_state_dict(torch.load(self.dest, map_location=cpu_device))
        self.model.load_state_dict(torch.load(self.dest))
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
