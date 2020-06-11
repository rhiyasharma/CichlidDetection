import os, shutil
from CichlidDetection.Classes.FileManagers import FileManager, ProjectFileManager
from CichlidDetection.Utilities.utils import area
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPrepper:
    """class to handle download and initial preparation of data required for training for faster_RCNN network
    """
    def __init__(self):
        self.file_manager = FileManager()
        self.proj_file_managers = {}

    def download_all(self):
        for pid in self.file_manager.unique_pids:
            self.proj_file_managers.update({pid: ProjectFileManager(pid, self.file_manager)})

    def prep(self):
        if not self.proj_file_managers:
            self.download_all()
        good_images = self._prep_labels()
        self._prep_images(good_images)
        self._generate_train_test_lists()

    def _prep_labels(self):
        # load the boxed fish csv
        df = pd.read_csv(self.file_manager.local_files['boxed_fish_csv'], index_col=0)
        # drop empty frames, frames labeled u, and incorrectly annotated frames
        df = df[(df.Nfish != 0) & ((df.Sex == 'm') | (df.Sex == 'f')) & (df.CorrectAnnotation == 'Yes')]
        # drop annotation boxes outside the area defined by the video points numpy
        df['Box'] = df['Box'].apply(eval)
        poly_vps = {}
        for pfm in self.proj_file_managers.values():
            poly_vps.update({pfm.pid: Polygon([list(row) for row in list(np.load(pfm.local_files['video_points_numpy']))])})
        df['Area'] = df.apply(lambda row: area(row, poly_vps), axis=1)
        df = df.dropna(subset=['Area'])
        # convert the 'Box' tuples to min and max x and y coordinates
        df[['xmin', 'ymin', 'w', 'h']] = pd.DataFrame(df['Box'].tolist(), index=df.index)
        df['xmax'] = df.xmin + df.w
        df['ymax'] = df.ymin + df.h
        df['label'] = [1 if sex == 'f' else 2 for sex in df.Sex]
        # trim down to only the required columns
        df = df.set_index('Framefile')
        df = df[['xmin', 'ymin', 'xmax', 'ymax', 'label']]
        # write a labelfile for each training image
        good_images = df.index.unique()
        for f in good_images:
            dest = os.path.join(self.file_manager.local_files['label_dir'], f.replace('.jpg', '.txt'))
            df.loc[[f]].to_csv(dest, sep=' ', header=False, index=False)
        return good_images

    def _prep_images(self, good_images):
        dest = self.file_manager.local_files['image_dir']
        for pid in self.file_manager.unique_pids:
            proj_image_dir = self.proj_file_managers[pid].local_files['project_image_dir']
            proj_images = [img for img in good_images if pid in img]
            for fname in proj_images:
                path = os.path.join(proj_image_dir, fname)
                shutil.copy(path, dest)

    def _generate_train_test_lists(self, train_size=0.8, random_state=42):
        img_dir = self.file_manager.local_files['image_dir']
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        train_files, test_files = train_test_split(img_files, train_size=train_size, random_state=random_state)
        with open(self.file_manager.local_files['train_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in train_files)
        with open(self.file_manager.local_files['test_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in test_files)
