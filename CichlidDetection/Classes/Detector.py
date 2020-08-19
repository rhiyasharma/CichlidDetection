import os
import time
from time import ctime
import pandas as pd
import numpy as np
import torch
import torchvision
from CichlidDetection.Classes.DataSet import DataSet, DetectDataSet, DetectVideoDataSet
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.ml_utils import collate_fn, Compose, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


class Detector:

    def __init__(self, *args):
        # initialize detector
        for i in args:
            self.pfm = i
        self.fm = FileManager()
        self._initiate_model()

    def test(self, n_imgs):
        """run detection on a random set of n images from the test set.
    
        Args:
        num: number of images to select

        """
        num = 'test_{}'.format(n_imgs)
        print("Start Time: ", ctime(time.time()))
        test_dataset = DataSet(Compose([ToTensor()]), 'test')
        indices = list(range(len(test_dataset)))
        np.random.shuffle(indices)
        idx = indices[:n_imgs]
        sampler = SubsetRandomSampler(idx)
        loader = DataLoader(test_dataset, sampler=sampler, batch_size=n_imgs, collate_fn=collate_fn)
        self.evaluate(loader, num)
        print("End Time: ", ctime(time.time()))

    def detect(self, img_dir):
        """run detection on the images contained in img_dir

        Args:
            img_dir (str): path to the image directory, relative to data_dir (see FileManager)
        """

        pid = img_dir.split('/')[1]
        img_dir = os.path.join(self.fm.local_files['data_dir'], img_dir)
        assert os.path.exists(img_dir)
        img_files = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)]
        dataset = DetectDataSet(Compose([ToTensor()]), img_files)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                collate_fn=collate_fn)
        self.evaluate(dataloader, pid)

    def frame_detect(self, pid, path, start, end):
        """run detection on the frame

        Args:
            path (str): path to the video directory (see ProjectFileManager)
        """
        video_name = path.split('/')[-1].split('.')[0] + '_{}'.format(start)
        print('beginning loading')
        dataset = DetectVideoDataSet(Compose([ToTensor()]), path, start, end, self.pfm)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                collate_fn=collate_fn)
        print('done loading')
        self.evaluate(dataloader, "{}_{}".format(pid, video_name))

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
    def evaluate(self, dataloader: DataLoader, name):
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

        if 'test' in name:
            df.to_csv(os.path.join(self.fm.local_files['detection_dir'], '{}_detections.csv'.format(name)))
        elif 'vid' in name:
            df.to_csv(os.path.join(self.fm.local_files['detection_dir'], '{}_detections.csv'.format(name)))
        else:
            df.to_csv(os.path.join(self.fm.local_files['detect_dir'], '{}_detections.csv'.format(name)))

        return '{}_detections.csv'.format(name)
