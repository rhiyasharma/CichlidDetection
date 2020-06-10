import csv
from shapely.geometry import Polygon
import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def collate_fn(batch):
    return tuple(zip(*batch))


def area(row, poly_vps):
    """
    """
    print(row.keys())
    x_a, y_a, w_a, h_a = row['Box']
    poly_ann = Polygon([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]])
    intersection_area = poly_ann.intersection(poly_vps[row['ProjectID']]).area
    ann_area = poly_ann.area
    return ann_area if ann_area == intersection_area else np.nan

def read_label_file(path):
    boxes = []
    labels = []
    with open(path) as f:
        for line in f.readlines():
            values = line.split()
            boxes.append([float(val) for val in values[:4]])
            labels.append(int(values[4]))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return {'boxes': boxes, 'labels': labels}

