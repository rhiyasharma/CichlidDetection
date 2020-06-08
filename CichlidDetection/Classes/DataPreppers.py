import os, shutil
import cv2

from CichlidDetection.Classes.FileManagers import FileManager
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pdb


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
    def __init__(self, fm):
        """initializes the DataPrepper for a particular pid, and downloads the required files from dropbox"""
        self.fm = fm
        self.pid = self.fm.pid
        if self.fm.pid is not None:
            self.fm.download_all()

    def prep_annotations(self):
        """Takes the BoxedFish.csv file and runs the necessary calculations to produce the CorrectAnnotations.csv file
        :return df: pandas dataframe corresponding to CorrectAnnotations.csv"""
        df = pd.read_csv(self.fm.local_files['boxed_fish_csv'], index_col=0)
        df = df[(df['ProjectID'] == self.pid) & (df['CorrectAnnotation'] == 'Yes') & (df['Sex'] != 'u')]
        empties = df[df['Box'].isnull()]
        df = df.dropna(subset=['Box'])
        df['Box'] = df['Box'].apply(eval)
        poly_vp = Polygon([list(row) for row in list(np.load(self.fm.local_files['video_points_numpy']))])
        df['Area'] = df['Box'].apply(lambda box: area(poly_vp, box))
        df = df.dropna(subset=['Area'])
        df = pd.concat([df, empties]).reset_index(drop=True)

        self.fm.local_files.update({'correct_annotations_csv': os.path.join(self.fm.local_files['project_dir'], 'CorrectAnnotations.csv')})
        df.to_csv(self.fm.local_files['correct_annotations_csv'])
        return df

    def view(self):
        """
        """
        df = self.prep_annotations()
        framefiles = df.Framefile.unique().tolist()
        mask = np.logical_not(np.load(self.fm.local_files['video_crop_numpy']))
        for frame in framefiles:
            img = cv2.imread(os.path.join(self.fm.local_files['image_dir'], frame))
            img[mask] = 0
            cv2.imshow("Modified Frame: {}".format(frame), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _generate_darknet_labels(self):
        # define a function that takes a row of CorrectAnnotations.csv and derives the annotation information expected
        # by darknet
        def custom_apply(row, img_size=(1296, 972)):
            fname = row['Framefile'].replace('.jpg', '.txt')
            if pd.isna(row['Box']):
                return [fname, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
            else:
                box = eval(row['Box'])
                label = 0 if row['Sex'] == 'm' else 1
                w = box[2]/img_size[0]
                h = box[3]/img_size[1]
                x_center = (box[0]/img_size[0]) + (w/2)
                y_center = (box[1]/img_size[1]) + (h/2)
                return [fname, label, x_center, y_center, w, h]

        # apply the custom_apply function to the dataframe, and use the resulting dataframe to iteratively create
        # a txt label file for each image
        df = pd.read_csv(self.fm.local_files['correct_annotations_csv'])
        df = df.apply(custom_apply, result_type='expand', axis=1).set_index(0)
        for f in df.index.unique():
            if df.loc[f].notna().all().all():
                dest = os.path.join(self.fm.local_files['label_dir'], f)
                df.loc[[f]].to_csv(dest, sep=' ', header=False, index=False)

    def _generate_train_test_lists(self, train_size=0.8, random_state=42):
        img_dir = self.fm.local_files['image_dir']
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        train_files, test_files = train_test_split(img_files, train_size=train_size, random_state=random_state)
        self.fm.local_files.update({'train_list': os.path.join(self.fm.local_files['training_dir'], 'train_list.txt'),
                                    'test_list': os.path.join(self.fm.local_files['training_dir'], 'test_list.txt')})
        with open(self.fm.local_files['train_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in train_files)
        with open(self.fm.local_files['test_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in test_files)

    def _generate_namefile(self):
        name_file = os.path.join(self.fm.local_files['training_dir'], 'CichlidDetection.names')
        if not os.path.exists(name_file):
            self.fm.local_files.update({'name_file': name_file})
            with open(self.fm.local_files['name_file'], 'w') as f:
                f.writelines('{}\n'.format(sex) for sex in ['male', 'female'])

    def _generate_datafile(self):
        data_file = os.path.join(self.fm.local_files['training_dir'], 'CichlidDetection.data')
        self.fm.local_files.update({'data_file': data_file})
        if not os.path.exists(data_file):
            fields = ['classes', 'train', 'valid', 'names']
            values = [2] + [self.fm.local_files[key] for key in ['train_list', 'test_list', 'name_file']]
            with open(self.fm.local_files['data_file'], 'w') as f:
                f.writelines('{}={}\n'.format(f, v) for (f, v) in list(zip(fields, values)))
            
class DataPrepper:
    """class to handle download and initial preparation of data required for training for faster_RCNN network
    """
    def __init__(self):
        """initializes the DataPrepper for a particular pid, and downloads the required files from dropbox"""
        fm = FileManager()
        cloud_files = fm.locate_cloud_files()
        self.local_files = {}
        self.local_files.update({'boxed_fish_csv_path':temp.download('boxed_fish_csv', cloud_files['boxed_fish_csv'])})
        self.master_dir = '/'.join(self.local_files['boxed_fish_csv_path'].split('/')[:-2])
        
    def download(self):
        df = pd.read_csv(self.local_files['boxed_fish_csv_path'], index_col=0)
        self.unique_pids = df.ProjectID.unique()
        for pid in self.unique_pids:
            fm = FileManager(pid)
            fm.download_images()
    
    def generate_train_validation_lists(self,train_size = 0.8, random_state=29):
        
        df = pd.read_csv(self.local_files['boxed_fish_csv_path'])
        self.local_files.update({'train_list': os.path.join(self.master_dir, 'train_list.txt'),
                                    'test_list': os.path.join(self.master_dir, 'test_list.txt')})
        df_subset = df[(df.Nfish != 0)&((df.Sex == 'm')|(df.Sex == 'f'))&(df.CorrectAnnotation=='Yes')]
        df_subset = df_subset.groupby(['ProjectID', 'Framefile']).size()
        indexs = df_subset.index
        img_files = []
        for project,frame in indexs:
            key = os.path.join(project,'images',frame)
            img_files.append(key)
        
        train_files, test_files = train_test_split(img_files, train_size=train_size, random_state=random_state)

        with open(self.local_files['train_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in train_files)
        with open(self.local_files['test_list'], 'w') as f:
            f.writelines('{}\n'.format(f_) for f_ in test_files)
            
        

    def _move_images(self):
        good_files = pd.read_csv(self.fm.local_files['correct_annotations_csv'])['Framefile'].to_list()
        for file in good_files:
            shutil.copy(os.path.join(self.fm.local_files['project_image_dir'], file), self.fm.local_files['image_dir'])
        return good_files

    def _cleanup(self):
        shutil.rmtree(self.fm.local_files['project_dir'])

