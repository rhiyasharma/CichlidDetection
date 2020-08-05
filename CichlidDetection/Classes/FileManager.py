import os
from itertools import chain
from os.path import join
import pandas as pd
from CichlidDetection.Utilities.utils import run, make_dir


class FileManager:
    """Project non-specific class for handling local and cloud storage."""

    def __init__(self):
        """create an empty local_files variable and run self._initialize()"""
        self.local_files = {}
        self._initialize()

    def sync_training_dir(self, exclude=None, quiet=False):
        """sync the training directory bidirectionally, keeping the newer version of each file

        Args:
            exclude (list of str): files/directories to exclude. Accepts both explicit file/directory names and
                regular expressions. Expects a list, even if it's a list of length one. Default None.
            quiet: if True, suppress the output of rclone copy. Default False
        """
        print('syncing training directory')
        down = ['rclone', 'copy', '-u', '-c', self.cloud_training_dir, self.local_files['training_dir']]
        up = ['rclone', 'copy', '-u', '-c', self.local_files['training_dir'], self.cloud_training_dir, '--exclude',
              '.*{/**,}']
        if not quiet:
            [com.insert(3, '-P') for com in [down, up]]
        if exclude is not None:
            [com.extend(list(chain.from_iterable(zip(['--exclude'] * len(exclude), exclude)))) for com in [down, up]]
        [run(com) for com in [down, up]]

    def _initialize(self):
        """create a required local directories, download essential files, and set the path for files generated later."""
        # create the any required directories that do not already exist
        self._make_dir('data_dir', join(os.getenv('HOME'), 'scratch', 'CichlidDetection'))
        self._make_dir('training_dir', join(self.local_files['data_dir'], 'training'))
        self._make_dir('train_image_dir', join(self.local_files['training_dir'], 'train_images'))
        self._make_dir('test_image_dir', join(self.local_files['training_dir'], 'test_images'))
        self._make_dir('label_dir', join(self.local_files['training_dir'], 'labels'))
        self._make_dir('log_dir', join(self.local_files['training_dir'], 'logs'))
        self._make_dir('weights_dir', join(self.local_files['training_dir'], 'weights'))
        self._make_dir('predictions_dir', join(self.local_files['training_dir'], 'predictions'))
        self._make_dir('figure_dir', join(self.local_files['training_dir'], 'figures'))
        self._make_dir('figure_data_dir', join(self.local_files['figure_dir'], 'figure_data'))
        self._make_dir('detection_dir', join(self.local_files['data_dir'], 'detection'))
        # locate and download remote files
        self.cloud_master_dir, cloud_files = self._locate_cloud_files()
        self.cloud_training_dir = join(self.cloud_master_dir, '___Tucker', 'CichlidDetection', 'training')

        for name, file in cloud_files.items():
            self._download(name, file, self.local_files['training_dir'])
        # set the paths of files that will be generated later
        for name, fname in [('train_list', 'train_list.txt'), ('test_list', 'test_list.txt')]:
            self.local_files.update({name: join(self.local_files['training_dir'], fname)})
        for name, fname in [('train_log', 'train.log'), ('batch_log', 'train_batch.log'), ('val_log', 'val.log')]:
            self.local_files.update({name: join(self.local_files['log_dir'], fname)})
        for name, fname in [('weights_file', 'last.weights')]:
            self.local_files.update({name: join(self.local_files['weights_dir'], fname)})
        for name, fname in [('ground_truth_csv', 'ground_truth.csv')]:
            self.local_files.update({name: join(self.local_files['predictions_dir'], fname)})
        # determine the unique project ID's from boxed_fish.csv
        self.unique_pids = pd.read_csv(self.local_files['boxed_fish_csv'], index_col=0)['ProjectID'].unique()

    def _download(self, name, source, destination_dir, overwrite=False):
        """use rclone to download a file, untar if it is a .tar file, and update self.local_files with the file path

        Args:
            name: brief descriptor of the file, used as the filepath key in self.local_files
            source: full path to a dropbox file, including the remote
            destination_dir: full path to the local destination directory
            overwrite: if True, run rclone copy even if a local file with the intended name already exists

        Returns:
            the full path to the newly downloaded file (or directory, if the file was a tarfile)
        """

        local_path = join(destination_dir, os.path.basename(source))
        if not os.path.exists(local_path) or overwrite:
            run(['rclone', 'copy', source, destination_dir])
            assert os.path.exists(local_path), "download failed\nsource: {}\ndestination_dir: {}".format(source,
                                                                                                         destination_dir)
        if os.path.splitext(local_path)[1] == '.tar':
            if not os.path.exists(os.path.splitext(local_path)[0]):
                run(['tar', '-xvf', local_path, '-C', os.path.dirname(local_path)])
            local_path = os.path.splitext(local_path)[0]
        self.local_files.update({name: local_path})
        return local_path

    def _locate_cloud_files(self):
        """locate the required files in Dropbox.

        Returns:
            string: cloud_master_dir, the outermost Dropbox directory that will be used henceforth
            dict: cloud_files, a dict of paths to remote files, keyed by brief descriptors
        """
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
        """update the self.local_files dict with {name: path}, and create the directory if it does not exist

        Args:
            name (str): brief file descriptor, to be used as key in the local_files dict
            path (str): local path of the directory to be created

        Returns:
            str: the path argument, unaltered
        """
        self.local_files.update({name: make_dir(path)})
        return path


class ProjectFileManager(FileManager):
    """Project specific class for managing local and cloud storage. Inherits from FileManager"""

    def __init__(self, pid, file_manager=None, download_images=False, download_videos=False, *video_names):
        """initialize a new FileManager, unless an existing file manager was passed to the constructor to save time

        Args:
            pid (str): project id
            file_manager (FileManager): optional. pass a pre-existing FileManager object to improve performance when
                initiating numerous ProjectFileManagers
            download_images (bool): if True, download the full image directory for the specified project
            download_videos (bool): if True, download the mp4 file in Videos directory of the specified project
            video_num (int): specifies which video to download
        """
        self.download_images = download_images
        self.download_videos = download_videos
        self.video_names = []
        for video in video_names:
            self.video_names.append(video)

        # initiate the FileManager parent class unless the optional file_manager argument is used
        if file_manager is None:
            FileManager.__init__(self)
        # if the file_manager argument is used, manually inherit the required attributes
        else:
            self.local_files = file_manager.local_files.copy()
            self.cloud_master_dir = file_manager.cloud_master_dir
        self.pid = pid
        # initialize project-specific directories
        self._initialize()

    def _initialize(self):
        """create project-specific directories and download project-specific files.

        Overwrites FileManager._initialize() method
        """
        self._make_dir('{}_dir'.format(self.pid), join(self.local_files['data_dir'], self.pid))

        if self.download_images:
            for name, file in self._locate_cloud_files().items():
                self._download(name, file, self.local_files['{}_dir'.format(self.pid)])

        if self.download_videos:
            for vid in self.video_names:
                for name, file in self._locate_cloud_files().items():
                    if name == vid:
                        self._download(name, file, self.local_files['{}_dir'.format(self.pid)])

    def _locate_cloud_files(self):
        """track down project-specific files in Dropbox.

        Overwrites FileManager._locate_cloud_files() method

        Returns:
            dict: cloud file paths keyed by brief file descriptors.
        """

        cloud_image_dir = join(self.cloud_master_dir, '__AnnotatedData/BoxedFish/BoxedImages/{}.tar'.format(self.pid))
        cloud_files = {'project_image_dir': cloud_image_dir} if self.download_images else {}
        remote_files = run(['rclone', 'lsf', join(self.cloud_master_dir, self.pid)])
        if 'videoCropPoints.npy' and 'videoCrop.npy' in remote_files.split():
            cloud_files.update({'video_points_numpy': join(self.cloud_master_dir, self.pid, 'videoCropPoints.npy')})
            cloud_files.update({'video_crop_numpy': join(self.cloud_master_dir, self.pid, 'videoCrop.npy')})
        else:
            cloud_files.update(
                {'video_points_numpy': join(self.cloud_master_dir, self.pid, 'MasterAnalysisFiles', 'VideoPoints.npy')})
            cloud_files.update(
                {'video_crop_numpy': join(self.cloud_master_dir, self.pid, 'MasterAnalysisFiles', 'VideoCrop.npy')})

        if self.download_videos:
            cloud_video_dir = join(self.cloud_master_dir, self.pid, 'Videos')
            videos_remote = run(['rclone', 'lsf', cloud_video_dir, '--include', '*.mp4']).split()
            for v in videos_remote:
                cloud_files.update({'{}'.format(v): join(self.cloud_master_dir, self.pid, 'Videos', v)})

        return cloud_files
