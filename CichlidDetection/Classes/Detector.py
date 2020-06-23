import os
import pandas as pd
import numpy as np
import torch
import torchvision
from CichlidDetection.Classes.DataSet import DataSet
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.utils import collate_fn, Compose, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

# Run this script from the main CichlidDetection directory: python3 Detector.py
# This script selects random images from the test set and runs them through the model to produce predictions 
# Results csv file saved in scratch/CichlidDetection/training/predictions as "Detect_images.csv"


class Detector:

    def __init__(self):
        # initialize detector
        self.fm = FileManager()
        self._initiate_model()

    def test(self, n_imgs):
        """run detection on a random set of n images from the test set.
    
        Args:
        num: number of images to select

        """
        dataset = DataSet(Compose([ToTensor()]), 'test')
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        idx = indices[:n_imgs]
        sampler = SubsetRandomSampler(idx)
        loader = DataLoader(dataset, sampler=sampler, batch_size=n_imgs, collate_fn=collate_fn)
        self.evaluate(loader)

    def detect(self, img_dir):
        """run detection on the images contained in img_dir

        Args:
            img_dir (str): path to the image directory
        """
        img_dir = os.path.join(self.fm.local_files['data_dir'], img_dir)
        assert os.path.exists(img_dir)


        pass

    def _initiate_model(self):
        """initiate the model, optimizer, and scheduler."""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.load_state_dict(torch.load(self.fm.local_files['weights_file']))
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.model.load_state_dict(torch.load(self.fm.local_files['weights_file'], map_location=self.device))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        """evaluate the model on the detect set of images"""
        cpu_device = torch.device("cpu")
        self.model.eval()
        results = {}
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        df = pd.DataFrame.from_dict(results, orient='index')
        index_list = df.index.tolist()
        detect_framefiles = []
        for i in index_list:
            detect_framefiles.append(dataloader.dataset.img_files[i])
        df['Framefile'] = [os.path.basename(path) for path in detect_framefiles]
        df = df[['Framefile', 'boxes', 'labels', 'scores']].set_index('Framefile')
        df.to_csv(os.path.join(self.fm.local_files['detection_dir'], 'detections.csv'))
