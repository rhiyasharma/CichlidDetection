import os
import numpy as np
import time
from PIL import Image

import pdb

from Utilities.utils import Logger,AverageMeter
import Utilities.transforms as T

from Classes.DataLoader.FasterRCNN_loader import CicilidDataset
from Classes.DataPrepper import FRCNN_DataPrepper

import torch
import torchvision
from torch.optim import lr_scheduler
from torch import optim


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
    

def train_epoch(epoch, data_loader, model,  optimizer, epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()
    for i,sth in enumerate(data_loader):
        print(i)
    
    
    
    for i, (images,targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        

        losses.update(losses.item(), inputs.size(0))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              .format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

#     if epoch % opt.checkpoint == 0:
#         save_file_path = os.path.join(opt.result_path,
#                                       'save_{}.pth'.format(epoch))

def main():
    
    dp = FRCNN_DataPrepper()
#     dp.download()
#     dp.generate_train_validation_lists()
    
    train_dataset = CicilidDataset(dp.master_dir, get_transform(train=True),'training')
    
#     for i in range(train_dataset.__len__()):
#         try:
#             img,boxes = train_dataset.__getitem__(i)
#             this_size = str(img.shape)
#             if this_size != default:
#                 print(this_size)
# 
#         except:
#             pdb.set_trace()
#             train_dataset.__getitem__(i)
#             print(i)
#             print(train_dataset.imgs[i])
#             print(train_dataset.boxes[train_dataset.imgs[i]])
#             break
    test_dataset = CicilidDataset(dp.master_dir, get_transform(train=False),'test')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)
    parameters = model.parameters()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    

# define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, num_workers=2,pin_memory=True,collate_fn=utils.collate_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False, num_workers=2,pin_memory=True,collate_fn=utils.collate_fn)

    optimizer = optim.SGD(parameters, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5)
            
            
    # create logger files
    train_logger = Logger(
            os.path.join(dp.master_dir, 'train.log'),
            ['epoch', 'loss', 'lr'])
    train_batch_logger = Logger(
            os.path.join(dp.master_dir, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'lr'])
    
    val_logger = Logger(
            os.path.join(dp.master_dir, 'val.log'), ['epoch', 'loss'])
    # let's train it for 10 epochs
    num_epochs = 10
    
    pdb.set_trace()
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
#         train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        train_epoch(epoch, train_loader, model, optimizer,train_logger, train_batch_logger)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
#         evaluate(model, data_loader_test, device=device)
    print("Done!")

if __name__ == "__main__":
    main()
