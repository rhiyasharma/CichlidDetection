import os, math, sys
import time
import pandas as pd
import torch
import torchvision
import random
import csv
from torchvision.transforms import functional as F
from CichlidDetection.Classes.DataLoader import DataLoader
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.torch_utils import MetricLogger, SmoothedValue, reduce_dict


def collate_fn(batch):
    """package a mini-batch of images and targets.

    Args:
        batch (list): uncollated mini-batch

    Returns:
        tuple: collated mini-batch
    """
    return tuple(zip(*batch))


class Trainer:
    """class to coordinate model training and evaluation"""

    def __init__(self, num_epochs, compare_annotations=True, upload_results=True):
        """initialize trainer

        Args:
            num_epochs (int): number of epochs to train
            compare_annotations: If True, evaluate the model on the test set after each epoch. This does not affect the
                end result of training, but does produce more data about model performance at each epoch. Setting to
                True also increases total runtime significantly
            upload_results: if True, automatically upload the results (weights, logs, etc.) after training
        """
        self.compare_annotations = compare_annotations
        self.fm = FileManager()
        self.num_epochs = num_epochs
        self.upload_results = upload_results
        self._initiate_loaders()
        self._initiate_model()

    def train(self):
        """train the model for the specified number of epochs."""
        for epoch in range(self.num_epochs):
            loss = self._train_epoch(epoch)
            self.scheduler.step(loss)
            if self.compare_annotations:
                self._evaluate_epoch(epoch)
        self._save_model()
        if self.upload_results:
            self.fm.sync_training_dir()

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

    def _get_transform(self, train):
        """get a composition of the appropriate data transformations.

        Args:
            train (bool): True if training the model, False if evaluating/testing the model

        Returns:
            composition of required transforms
        """
        transforms = [ToTensor()]
        if train:
            transforms.append(RandomHorizontalFlip(0.5))
        return Compose(transforms)

    def _train_epoch(self, epoch):
        """train the model for one epoch.

        Args:
            epoch (int): epoch number, greater than or equal to 0

        Returns:
            float: averaged epoch loss
        """
        print('train at epoch {}'.format(epoch))
        self.model.train()
        metric_logger = MetricLogger(delimiter=', ', f=self.fm.local_files['train_log'])
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(self.train_loader, print_freq=1, header=header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        return metric_logger

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


# helper classes


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


class AverageMeter(object):
    """Computes and stores the running average of a metric. Useful for updating metrics after each epoch / batch."""

    def __init__(self):
        """default all values to upon class declaration."""
        self._reset()

    def _reset(self):
        """reset all metrics to 0 when initiating."""
        self.val = 0    #: current value
        self.avg = 0    #: running average of metric
        self.sum = 0    #: running sum of metric
        self.count = 0  #: running count of metric

    def update(self, val, n=1):
        """Update the current value, as well as the running sum, count, and average.

        Args:
            val (float): value used to update metrics
            n (int): number of instances with the value val. Default 1

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


