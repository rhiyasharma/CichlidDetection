import os, shutil
from os.path import join
import pandas as pd
from CichlidDetection.Utilities.system_utilities import run, make_dir


class FileManager:
    """Project non-specific class for setting up local directories, downloading required files, and keeping track of local file paths"""

    def __init__(self):
        self.local_files = {}
        self._initialize()

    def _initialize(self):
        """create a required local directories if they do not already exist, downloads a few essential files,
        and sets the path to a few files that will be created later"""
        self._make_dir('data_dir', join(os.getenv('HOME'), 'scratch', 'CichlidDetection'))
        self._make_dir('training_dir', join(self.local_files['data_dir'], 'training'))
        self._make_dir('image_dir', join(self.local_files['training_dir'], 'images'))
        self._make_dir('label_dir', join(self.local_files['training_dir'], 'labels'))
        self._make_dir('log_dir', join(self.local_files['training_dir'], 'logs'))
        self._make_dir('weights_dir', join(self.local_files['training_dir'], 'weights'))
        self.cloud_master_dir, cloud_files = self._locate_cloud_files()
        for name, file in cloud_files.items():
            self._download(name, file, self.local_files['training_dir'])
        for name, fname in [('train_list', 'train_list.txt'), ('test_list', 'test_list.txt'),
                            ('box_predictions_csv', 'box_predictions.csv'),
                            ('label_predictions_csv', 'label_predicitons.csv')]:
            self.local_files.update({name: join(self.local_files['training_dir'], fname)})
        for name, fname in [('train_log', 'train.log'), ('batch_log', 'train_batch.log'), ('val_log', 'val.log')]:
            self.local_files.update({name: join(self.local_files['log_dir'], fname)})
        for name, fname in [('weights_file', 'last.weights')]:
            self.local_files.update({name: join(self.local_files['weights_dir'], fname)})
        self.unique_pids = pd.read_csv(self.local_files['boxed_fish_csv'], index_col=0)['ProjectID'].unique()


    def _download(self, name, source, destination_dir, overwrite=False):
        """use rclone to download a file, and untar if it is a .tar file. Automatically adds file path to self.local_files
        :param name: brief descriptor of the file, to be used for easy access to the file path using the self.local_files dict
        :param source: full path to a dropbox file, including the remote
        :param destination: full path to the local destination directory
        :param overwrite: if True, run rclone copy even if a local file with the intended name already exists
        :return local_path: the full path the to the newly downloaded file (or directory, if the file was a tarfile)
        """

        local_path = join(destination_dir, os.path.basename(source))
        if not os.path.exists(local_path) or overwrite:
            run(['rclone', 'copy', source, destination_dir])
            assert os.path.exists(local_path), "download failed\nsource: {}\ndestination_dir: {}".format(source, destination_dir)
        if os.path.splitext(local_path)[1] == '.tar':
            if not os.path.exists(os.path.splitext(local_path)[0]):
                run(['tar', '-xvf', local_path, '-C', os.path.dirname(local_path)])
            local_path = os.path.splitext(local_path)[0]
        self.local_files.update({name: local_path})
        return local_path

    def _locate_cloud_files(self):
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
        cloud_master_dir = join(remote + root_dir, 'Apps', 'CichlidPiData')

        # locate essential, project non-specific files
        cloud_files = {'boxed_fish_csv': join(cloud_master_dir, '__AnnotatedData/BoxedFish/BoxedFish.csv')}

        return cloud_master_dir, cloud_files

    def _make_dir(self, name, path):
        self.local_files.update({name: make_dir(path)})
        return path


class ProjectFileManager(FileManager):
    """Project specific class for setting up local directories, downloading required files, and keeping track of local file paths"""

    def __init__(self, pid, file_manager=None):
        # initialize the parent class, unless an existing file manager was passed to the constructor to save time
        if file_manager is None:
            FileManager.__init__(self)
        else:
            self.local_files = file_manager.local_files.copy()
            self.cloud_master_dir = file_manager.cloud_master_dir
        self.pid = pid
        self._initialize()

    def _initialize(self):
        """create a required local directories if they do not already exist and download project specific files if not present"""
        self._make_dir('project_dir', join(self.local_files['data_dir'], self.pid))
        for name, file in self._locate_cloud_files().items():
            self._download(name, file, self.local_files['project_dir'])

    def _locate_cloud_files(self):
        # track down the project-specific files with multiple possible names / locations
        cloud_image_dir = join(self.cloud_master_dir, '__AnnotatedData/BoxedFish/BoxedImages/{}.tar'.format(self.pid))
        cloud_files = {'project_image_dir': cloud_image_dir}
        remote_files = run(['rclone', 'lsf', join(self.cloud_master_dir, self.pid)])
        if 'videoCropPoints.npy' and 'videoCrop.npy' in remote_files.split():
            cloud_files.update({'video_points_numpy': join(self.cloud_master_dir, self.pid, 'videoCropPoints.npy')})
            cloud_files.update({'video_crop_numpy': join(self.cloud_master_dir, self.pid, 'videoCrop.npy')})
        else:
            cloud_files.update({'video_points_numpy': join(self.cloud_master_dir, self.pid, 'MasterAnalysisFiles', 'VideoPoints.npy')})
            cloud_files.update({'video_crop_numpy': join(self.cloud_master_dir, self.pid, 'MasterAnalysisFiles', 'VideoCrop.npy')})
        return cloud_files



