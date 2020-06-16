import os
import time
import pandas as pd
import torch
import torchvision
import random
from torchvision.transforms import functional as F

from CichlidDetection.Utilities.utils import Logger, AverageMeter, collate_fn
import CichlidDetection.Utilities.transforms as T
from CichlidDetection.Classes.DataLoader import DataLoader
from CichlidDetection.Classes.FileManager import FileManager


class Trainer:
    """class to coordinate model training and evaluation"""

    def __init__(self, num_epochs, compare_annotations=True):
        """initialize trainer

        Args:
            num_epochs (int): number of epochs to train
            compare_annotations: If True, evaluate the model on the test set after each epoch. This does not affect the
                end result of training, but does produce significantly more data about model performance at each epoch.
                Setting to True increases total runtime significantly
        """
        self.compare_annotations = compare_annotations
        self.fm = FileManager()
        self.num_epochs = num_epochs
        self._initiate_loaders()
        self._initiate_model()
        self._initiate_loggers()

    def train(self):
        """train the model for the specified number of epochs."""
        for epoch in range(self.num_epochs):
            loss = self._train_epoch(epoch)
            self.scheduler.step(loss)
            if self.compare_annotations:
                self._evaluate_epoch(epoch)
        self._save_model()

    def _initiate_loaders(self):
        """initiate train and test datasets and  dataloaders."""
        self.train_dataset = DataLoader(self._get_transform(train=True), 'train')
        self.test_dataset = DataLoader(self._get_transform(train=False), 'test')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=5, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
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

    def _initiate_loggers(self):
        """initiate loggers to track training progress."""
        self.train_logger = Logger(self.fm.local_files['train_log'], ['epoch', 'loss', 'lr'])
        self.train_batch_logger = Logger(self.fm.local_files['batch_log'], ['epoch', 'batch', 'iter', 'loss', 'lr'])
        self.val_logger = Logger(self.fm.local_files['val_log'], ['epoch'])

    def _get_transform(self, train):
        """get a composition of the appropriate data transformations.

        Args:
            train (bool): True if training the model, False if evaluating/testing the model

        Returns:
            composition of required transforms
        """
        transforms = [ToTensor()]
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def _train_epoch(self, epoch):
        """train the model for one epoch.

        Args:
            epoch (int): epoch number, greater than or equal to 0

        Returns:
            float: averaged epoch loss
        """
        print('train at epoch {}'.format(epoch))
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end_time = time.time()

        for i, (images, targets) in enumerate(self.train_loader):
            data_time.update(time.time() - end_time)
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            today_loss = sum(loss for loss in loss_dict.values())
            losses.update(today_loss.item(), len(images))

            self.optimizer.zero_grad()
            today_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            self.train_batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(self.train_loader) + (i + 1),
                'loss': losses.val,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch,
                      i + 1,
                      len(self.train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
        self.train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        return losses.avg

    @torch.no_grad()
    def _evaluate_epoch(self, epoch):
        """evaluate the model on the test set following an epoch of training.

        Args:
            epoch (int): epoch number, greater than or equal to 0

        """
        print('evaluating epoch {}'.format(epoch))
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
        df.to_csv(os.path.join(self.fm.local_files['predictions_dir'], '{}.csv'.format(epoch)))

    def _save_model(self):
        """save the weights file (state dict) for the model."""
        dest = self.fm.local_files['weights_file']
        if os.path.exists(dest):
            path = os.path.join(self.fm.local_files['weights_dir'], str(int(os.path.getmtime(dest))) + '.weights')
            os.rename(dest, path)
        torch.save(self.model.state_dict(), dest)


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


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


