import os, shutil
from CichlidDetection.Classes.FileManager import FileManager, ProjectFileManager
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


class DataPrepper:
    """class to handle the required data prep prior to training the model"""
    def __init__(self):
        """initiate a FileManager object, and and empty dictionary to store a ProjectFileManager object for each project"""
        self.file_manager = FileManager()
        self.proj_file_managers = {}

    def download_all(self):
        """initiate a ProjectFileManager for each unique project. This automatically downloads any missing files"""
        for pid in self.file_manager.unique_pids:
            self.proj_file_managers.update({pid: ProjectFileManager(pid, self.file_manager)})

    def prep(self):
        """prep the label files, image files, and train-test lists required for training"""
        if not self.proj_file_managers:
            self.download_all()
        good_images = self._prep_labels()
        self._prep_images(good_images)
        self._generate_train_test_lists()

    def _prep_labels(self):
        """generate a label file for each valid image

        valid images are those in the boxed fish csv for which CorrectAnnotation is Yes, Sex is m or f, Nfish > 0, and
        the annotation box falls entirely within the boundaries defined by the video points numpy file

        Returns:
            list: file names of the images valid for training/testing
        """
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
        # write a labelfile for each image remaining in df
        good_images = list(df.index.unique())
        for f in good_images:
            dest = os.path.join(self.file_manager.local_files['label_dir'], f.replace('.jpg', '.txt'))
            df.loc[[f]].to_csv(dest, sep=' ', header=False, index=False)
        return good_images

    def _prep_images(self, good_images):
        """copy valid images from individual project directories to a centralized image directory

        Args:
            good_images (list of str): file names of valid images to move
        """
        dest = self.file_manager.local_files['image_dir']  # centralized image dir
        for pid in self.file_manager.unique_pids:
            proj_image_dir = self.proj_file_managers[pid].local_files['project_image_dir']
            candidates = os.listdir(proj_image_dir)
            proj_images = [img for img in good_images if img in candidates]
            for fname in proj_images:
                path = os.path.join(proj_image_dir, fname)
                shutil.copy(path, dest)

    def _generate_train_test_lists(self, train_size=0.8, random_state=42):
        """split the valid images into training and testing sets, and write corresponding train list and test list files

        Args:
            train_size (float): proportion of data to use for training, 0 to 1
            random_state (int): random state seed for repeatability
        """
        img_dir = self.file_manager.local_files['image_dir']
        img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        train_files, test_files = train_test_split(img_files, train_size=train_size, random_state=random_state)
        with open(self.file_manager.local_files['train_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in train_files)
        with open(self.file_manager.local_files['test_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in test_files)

    def _generate_ground_truth_csv(self):
        """generate a csv of testing targets for comparison with the output of Trainers.Trainer._evaluate_epoch()"""
        # load the boxed fish csv and narrow to valid images
        df = pd.read_csv(self.file_manager.local_files['boxed_fish_csv'])
        df = df[(df.Nfish != 0) & ((df.Sex == 'm') | (df.Sex == 'f')) & (df.CorrectAnnotation == 'Yes')]
        # parse the test list from test_list.txt
        with open(self.file_manager.local_files['test_list']) as f:
            frames = [os.path.basename(frame) for frame in f.read().splitlines()]
        # narrow dataframe to images in the test list
        df = df.loc[df.Framefile.isin(frames), :][['Framefile', 'Box', 'Sex']]
        # coerce the values into the correct form
        df.Sex = df.Sex.apply(lambda x: 1 if x is 'f' else 2)
        df['Box'] = df['Box'].apply(eval).apply(list)
        df = df.groupby('Framefile').agg(lambda x: list(x))
        df.rename(columns={'Box': 'boxes', 'Sex': 'labels'}, inplace=True)
        df.to_csv(self.file_manager.local_files['ground_truth_csv'])

