import csv
from shapely.geometry import Polygon
import numpy as np
import torch


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


class Logger(object):
    """manages creation of logfiles that track basic training/evaluation stats."""

    def __init__(self, path, header):
        """open the logfile and write its header.

        Args:
            path (str): path to the logfile
            header (list of str): column names
        """
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        """close the logfile."""
        self.log_file.close()

    def log(self, values):
        """write a new row to the logfile.

        Args:
            values (dict): dictionary of key-value pairs, where each key must be a column name in self.header
        """
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def collate_fn(batch):
    """package a mini-batch of images and targets.

    Args:
        batch (list): uncollated mini-batch

    Returns:
        tuple: collated mini-batch
    """
    return tuple(zip(*batch))


def area(row, poly_vps):
    """calculate the annotation box area

    Args:
        row: pandas dataframe row containing, at minimum, the box coordinates (in x, y, w, h form) and project id
        poly_vps (dict): dictionary of video crops as Shapely Polygons, keyed by project id

    Returns:
        float: annotation area if the box is within the video crop boundaries, else np.nan
    """
    x_a, y_a, w_a, h_a = row['Box']
    poly_ann = Polygon([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]])
    intersection_area = poly_ann.intersection(poly_vps[row['ProjectID']]).area
    ann_area = poly_ann.area
    return ann_area if ann_area == intersection_area else np.nan


def read_label_file(path):
    """read box coordinates and labels from the label file and return them as a target dictionary.

    Args:
        path (str): path to label file

    Returns:
        dict: target dictionary containing the boxes tensor and labels tensor
    """
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
