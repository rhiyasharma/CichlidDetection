import os, shutil
import pandas as pd
from CichlidDetection.Utilities.SystemUtilities import run, make_dir


class FileManager:
    """class for setting up local directories, downloading required files, and keeping track of local file paths"""

    def __init__(self, pid=None):
        self.pid = pid
        self.local_files = {}
        self.initialize()

    def initialize(self):
        """create a required local directories if they do not already exist"""
        data_dir = os.path.join(os.getenv('HOME'), 'scratch', 'CichlidDetection')
        training_dir = os.path.join(data_dir, 'training')
        image_dir = os.path.join(training_dir, 'images')
        label_dir = os.path.join(training_dir, 'labels')
        self.make_dir('training_dir', training_dir)
        self.make_dir('image_dir', image_dir)
        self.make_dir('label_dir', label_dir)
        if self.pid is not None:
            project_dir = os.path.join(data_dir, self.pid)
            self.make_dir('project_dir', project_dir)

    def get_all_pids(self):
        # intended to get all PIDS from boxed fish csv, but ended up hard coding it after running into incomplete
        # projects

        # source = self.locate_cloud_files()['boxed_fish_csv']
        # self.download(name='boxed_fish_csv', source=source, destination=self.local_files['training_dir'])
        # return pd.read_csv(self.local_files['boxed_fish_csv'], index_col=0)['ProjectID'].unique()

        return ['CV10_3', 'CV_fem_con1', 'CV_fem_con2', 'MC6_5', 'MC16_2', 'MC_fem_con1',
                'MC_fem_con2', 'MCxCVF1_12a_1', 'MCxCVF1_12b_1', 'TI2_4', 'TI3_3']

    def download_all(self):
        """downloads all files necessary to run DataPrepper.py"""
        cloud_files = self.locate_cloud_files()
        for name, file in cloud_files.items():
            self.download(name, file)

    def download(self, name, source, destination=None):
        """use rclone to download a file, and untar if it is a .tar file. Automatically adds file path to self.local_files
        :param name: brief descriptor of the file, to be used for easy access to the file path using the self.local_files dict
        :param source: full path to a dropbox file, including the remote
        :param destination: full path to the local destination directory. Defaults to self.project_dir
        :return local_path: the full path the to the newly downloaded file (or directory, if the file was a tarfile)
        """
        destination = self.local_files['project_dir'] if destination is None else destination
        local_path = os.path.join(destination, os.path.basename(source))
        if not os.path.exists(local_path):
            run(['rclone', 'copy', source, destination])
            assert os.path.exists(local_path), "download failed\nsource: {}\ndestination: {}".format(source, destination)
        if os.path.splitext(local_path)[1] == '.tar':
            if not os.path.exists(os.path.splitext(local_path)[0]):
                run(['tar', '-xvf', local_path, '-C', os.path.dirname(local_path)])
            local_path = os.path.splitext(local_path)[0]
        self.local_files.update({name: local_path})
        return local_path

    def locate_cloud_files(self):
        """locate the files in the cloud necessary to run PrepareTrainingData.py
        :return cloud_files: a dictionary of source paths of the form expected by FileManager.download(), indexed
        identically to the dictionary returned by download_all"""

        # establish the correct remote
        possible_remotes = run(['rclone', 'listremotes']).split()
        if len(possible_remotes) == 1:
            remote = possible_remotes[0]
        elif 'cichlidVideo:' in possible_remotes:
            remote = 'cichlidVideo:'
        elif 'd:' in possible_remotes:
            remote = 'd:'
        else:
            raise Exception('unable to establish rclone remote')

        # establish the correct path to the CichlidPiData directory
        root_dir = [r for r in run(['rclone', 'lsf', remote]).split() if 'McGrath' in r][0]
        base = os.path.join(remote + root_dir, 'Apps', 'CichlidPiData')

        # start the cloud_files dictionary with the easy to find files
        cloud_files = {'boxed_fish_csv': os.path.join(base, '__AnnotatedData/BoxedFish/BoxedFish.csv')}

        # track down the project-specific files with multiple possible names / locations
        if self.pid is not None:
            cloud_files.update({'project_image_dir': os.path.join(base, '__AnnotatedData/BoxedFish/BoxedImages/{}.tar'.format(self.pid))})
            remote_files = run(['rclone', 'lsf', os.path.join(base, self.pid)])
            if 'videoCropPoints.npy' and 'videoCrop.npy' in remote_files.split():
                cloud_files.update({'video_points_numpy': os.path.join(base, self.pid, 'videoCropPoints.npy')})
                cloud_files.update({'video_crop_numpy': os.path.join(base, self.pid, 'videoCrop.npy')})
            else:
                cloud_files.update({'video_points_numpy': os.path.join(base, self.pid, 'MasterAnalysisFiles', 'VideoPoints.npy')})
                cloud_files.update({'video_crop_numpy': os.path.join(base, self.pid, 'MasterAnalysisFiles', 'VideoCrop.npy')})

        return cloud_files

    def make_dir(self, name, path):
        self.local_files.update({name: make_dir(path)})
        return path









