import os
import pandas as pd
from collections import defaultdict
from PIL import Image
import torch

class CicilidDataset(object):
    def __init__(self, root, transforms,subset):
        self.root = root
        self.transforms = transforms
        self.boxes = defaultdict(list)
        self.labels = defaultdict(list)
        
        
        boxed_file = os.path.join(root,'None','BoxedFish.csv')
        if subset == 'training':
            self.files_list = os.path.join(root,'train_list.txt')
        elif subset == 'test':
            self.files_list = os.path.join(root,'test_list.txt')
        
        df = pd.read_csv(boxed_file)
        for index, row in df.iterrows():
            if row.Nfish == 0 or not row.Sex or row.Sex == 'u' or row.CorrectAnnotation == 'No':
                continue
            key = row.ProjectID+'/images/'+row.Framefile
            self.boxes[key].append(row.Box)
            if row.Sex == 'm':
                self.labels[key].append(0)
            else:
                self.labels[key].append(1)
        self.imgs = []
        with open(self.files_list,'r') as input:
            for line in input:
                self.imgs.append(line.rstrip())


    def __getitem__(self, idx):
        # load images ad masks
        key = self.imgs[idx]
        img_path = os.path.join(self.root, key)
        img = Image.open(img_path).convert("RGB")
        box_old = self.boxes[key]
        boxes = []

        for i,box in enumerate(box_old):
            box = [int(x) for x in box[1:-1].split(',')]
            xmin = box[0]
            xmax = box[0] + box[2]
            ymin = box[1]
            ymax = box[1] + box[3]
            boxes.append([xmin, ymin, xmax, ymax])
        labels = self.labels[key]
        assert len(labels) == len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
       
        labels = torch.as_tensor(labels, dtype=torch.int64)


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
#         target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
#         target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)