import os
import cv2
from CichlidDetection.Classes.FileManager import FileManager
from shapely.geometry import Polygon
import numpy as np
import pandas as pd


def area(poly_vp, box):
    """
    :param poly_vp:
    :param box:
    :return:
    """
    x_a, y_a, w_a, h_a = box
    poly_ann = Polygon([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]])
    intersection_area = poly_ann.intersection(poly_vp).area
    ann_area = poly_ann.area
    return ann_area if ann_area == intersection_area else np.nan


class DataPrepper:
    """class to handle the download and initial preparation of data required for training
    :param pid: short for ProjectID. The name of the project to be analyzed, for example, 'MC6_5'
    """
    def __init__(self, pid):
        """initializes the DataPrepper for a particular pid, and downloads the required files from dropbox"""
        self.pid = pid
        self.fm = FileManager(pid)
        self.fm.download_all()

    def prep(self):
        """Takes the BoxedFish.csv file and runs the necessary calculations to produce the CorrectAnnotations.csv file
        :return df: pandas dataframe corresponding to CorrectAnnotations.csv"""
        df = pd.read_csv(self.fm.local_files['boxed_fish_csv'], index_col=0)
        df = df[(df['ProjectID'] == self.pid) & (df['CorrectAnnotation'] == 'Yes')].dropna(subset=['Box'])
        df['Box'] = df['Box'].apply(eval)
        poly_vp = Polygon([list(row) for row in list(np.load(self.fm.local_files['video_points_numpy']))])
        df['Area'] = df['Box'].apply(lambda box: area(poly_vp, box))
        df = df.dropna(subset=['Area']).reset_index()

        self.fm.local_files.update({'correct_annotations_csv': os.path.join(self.fm.project_dir, 'CorrectAnnotations.csv')})
        df.to_csv(self.fm.local_files['correct_annotations_csv'])
        return df

    def view(self):
        """
        """
        df = self.prep()
        framefiles = df.Framefile.unique().tolist()
        mask = np.logical_not(np.load(self.fm.local_files['video_crop_numpy']))
        for frame in framefiles:
            img = cv2.imread(os.path.join(self.fm.local_files['image_folder'], frame))
            img[mask] = 0
            cv2.imshow("Modified Frame: {}".format(frame), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def generate_darknet_labels(self):
        df = self.prep()



