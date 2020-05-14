import os, shutil
import cv2
from CichlidDetection.Classes.FileManager import FileManager
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    def prep_annotations(self):
        """Takes the BoxedFish.csv file and runs the necessary calculations to produce the CorrectAnnotations.csv file
        :return df: pandas dataframe corresponding to CorrectAnnotations.csv"""
        df = pd.read_csv(self.fm.local_files['boxed_fish_csv'], index_col=0)
        df = df[(df['ProjectID'] == self.pid) & (df['CorrectAnnotation'] == 'Yes')].dropna(subset=['Box'])
        df['Box'] = df['Box'].apply(eval)
        poly_vp = Polygon([list(row) for row in list(np.load(self.fm.local_files['video_points_numpy']))])
        df['Area'] = df['Box'].apply(lambda box: area(poly_vp, box))
        df = df.dropna(subset=['Area']).reset_index(drop=True)

        self.fm.local_files.update({'correct_annotations_csv': os.path.join(self.fm.project_dir, 'CorrectAnnotations.csv')})
        df.to_csv(self.fm.local_files['correct_annotations_csv'])
        return df

    def view(self):
        """
        """
        df = self.prep_annotations()
        framefiles = df.Framefile.unique().tolist()
        mask = np.logical_not(np.load(self.fm.local_files['video_crop_numpy']))
        for frame in framefiles:
            img = cv2.imread(os.path.join(self.fm.local_files['image_folder'], frame))
            img[mask] = 0
            cv2.imshow("Modified Frame: {}".format(frame), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def YOLO_prep(self):
        self._generate_image_folder()
        self._generate_darknet_labels()
        self._generate_train_test_lists()
        self._generate_namefile()
        self._generate_datafile()

    def _generate_darknet_labels(self):
        # check the image size
        img_folder = self.fm.local_files['image_folder']
        test_img = next(os.path.join(img_folder, f) for f in os.listdir(img_folder))
        test_img_size = cv2.imread(test_img).shape
        test_img_size = (test_img_size[1], test_img_size[0])

        # create a folder for the label files
        label_folder = self.fm.make_dir('label_folder', os.path.join(self.fm.local_files['project_directory'], 'labels'))

        # define a function that takes a row of CorrectAnnotations.csv and derives the annotation information expected
        # by darknet
        def custom_apply(row, img_size):
            fname = row['Framefile'].replace('.jpg', '.txt')
            label = 0 if row['Sex'] == 'm' else 1
            w = row['Box'][2]/img_size[0]
            h = row['Box'][3]/img_size[1]
            x_center = (row['Box'][0]/img_size[0]) + (w/2)
            y_center = (row['Box'][1]/img_size[1]) + (h/2)
            return [fname, label, x_center, y_center, w, h]

        # apply the custom_apply function to the dataframe, and use the resulting dataframe to iteratively create
        # a txt label file for each image
        df = self.prep_annotations().apply(custom_apply, result_type='expand', axis=1, img_size=test_img_size).set_index(0)
        for f in df.index.unique():
            dest = os.path.join(self.fm.local_files['label_folder'], f)
            df.loc[[f]].to_csv(dest, sep=' ', header=False, index=False)

    def _generate_train_test_lists(self, train_size=0.8, random_state=42):
        img_dir = self.fm.local_files['image_folder']
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        train_files, test_files = train_test_split(img_files, train_size=train_size, random_state=random_state)
        self.fm.local_files.update({'train_list': os.path.join(self.fm.project_dir, 'train_list.txt'),
                                    'test_list': os.path.join(self.fm.project_dir, 'test_list.txt')})
        with open(self.fm.local_files['train_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in train_files)
        with open(self.fm.local_files['test_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in test_files)

    def _generate_namefile(self):
        self.fm.local_files.update({'name_file': os.path.join(self.fm.project_dir, 'CichlidDetection.names')})
        with open(self.fm.local_files['name_file'], 'w') as f:
            f.writelines('{}\n'.format(sex) for sex in ['male', 'female'])

    def _generate_datafile(self):
        fields = ['classes', 'train', 'valid', 'names']
        values = [2] + [self.fm.local_files[key] for key in ['train_list', 'test_list', 'name_file']]
        self.fm.local_files.update({'data_file': os.path.join(self.fm.project_dir, 'CichlidDetection.data')})
        with open(self.fm.local_files['data_file'], 'w') as f:
            f.writelines('{}={}\n'.format(f, v) for (f, v) in list(zip(fields, values)))

    def _generate_image_folder(self):
        good_files = pd.read_csv(self.fm.local_files['correct_annotations_csv'])['Framefile'].to_list()
        self.fm.make_dir('image_folder', os.path.join(self.fm.project_dir, 'images'))
        for file in good_files:
            shutil.copy(os.path.join(self.fm.local_files['all_image_folder'], file), self.fm.local_files['image_folder'])

