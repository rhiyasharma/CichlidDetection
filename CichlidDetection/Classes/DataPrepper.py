import os
from CichlidDetection.Classes.FileManager import FileManager
from shapely.geometry import Polygon
import numpy as np
import pandas as pd


def area(poly_vp, box):
    x_a, y_a, w_a, h_a = box
    poly_ann = Polygon([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]])
    intersection_area = poly_ann.intersection(poly_vp).area
    ann_area = poly_ann.area
    return intersection_area, ann_area


class DataPrepper:

    def __init__(self, pid):
        self.pid = pid
        self.fm = FileManager(pid)
        self.local_files = self.fm.download_all()

    def prep(self):
        df = pd.read_csv(self.local_files['boxed_fish_csv'])
        df = df[(df['ProjectID'] == self.pid) & (df['CorrectAnnotation'] == 'Yes')].dropna(subset=['Box'])
        poly_vp = Polygon([list(row) for row in list(np.load(self.local_files['video_points_numpy']))])



