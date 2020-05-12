import os
from CichlidDetection.Utilities.SystemUtilities import run, make_dir


class FileManager:

    def __init__(self, pid):
        self.pid = pid
        self.project_dir = self.initialize_project_directory()

    def initialize_project_directory(self):
        """create a local project directory if it does not already exist
        :return project_dir: the absolute local path to the project directory that was created
        """
        project_dir = os.path.join(os.getenv('HOME'), 'scratch', 'CichlidDetection', self.pid)
        make_dir(project_dir)
        return project_dir

    def download_all(self):
        """downloads all files necessary to run PrepareTrainingData.py
        :return local_files: dictionary of local file paths indexed by brief file descriptors"""
        cloud_files = self.locate_cloud_files()
        local_files = {}
        for name, file in cloud_files.items():
            local_files.update({name: self.download(file)})
        return local_files

    def download(self, source, destination=None):
        """use rclone to download a file, and untar if it is a .tar file
        :param source: full path to a dropbox file, including the remote
        :param destination: full path to the local destination directory. Defaults to self.project_dir
        :return local_path: the full path the to the newly downloaded file (or directory, if the file was a tarfile)
        """
        destination = self.project_dir if destination is None else destination
        run(['rclone', 'copy', source, destination])
        local_path = os.path.join(destination, os.path.basename(source))
        if os.path.splitext(local_path)[1] == '.tar':
            run(['tar', '-xvf', local_path, '-C', os.path.splitext(local_path)[0]])
            local_path = os.path.splitext(local_path)[0]
        assert os.path.exists(local_path), "download failed\nsource: {}\ndestination: {}".format(source, destination)
        return local_path

    def locate_cloud_files(self):
        """locate the files in the cloud necessary to run PrepareTrainingData.py
        :return cloud_files: a dictionary of source paths of the form expected by FileManager.download(), indexed
        identically to the dictionary returned by download_all"""

        # establish the correct path to the CichlidPiData directory
        remote = 'cichlidVideo:'
        root_dir = [r for r in run(['rclone', 'lsf', remote]).split() if 'McGrath' in r][0]
        base = os.path.join(remote + root_dir, 'Apps', 'CichlidPiData')

        # start the cloud_files dictionary with the easy to find files
        cloud_files = {'boxed_fish_csv': os.path.join(base, '__AnnotatedData/BoxedFish/BoxedFish.csv')}
        cloud_files.update({'image_folder': os.path.join(base, '__AnnotatedData/BoxedFish/BoxedImages/{}.tar'.format(self.pid))})

        # track down the project-specific files with multiple possible names / locations
        remote_files = run(['rclone', 'lsf', os.path.join(base, self.pid)])
        if 'videoCropPoints.npy' and 'videoCrop.npy' in remote_files.stdout.split():
            cloud_files.update({'video_points_numpy': os.path.join(base, self.pid, 'videoCropPoints.npy')})
            cloud_files.update({'video_crop_numpy': os.path.join(base, self.pid, 'videoCrop.npy')})
        else:
            cloud_files.update({'video_points_numpy': os.path.join(base, self.pid, 'MasterAnalysisFiles', 'VideoPoints.npy')})
            cloud_files.update({'video_crop_numpy': os.path.join(base, self.pid, 'MasterAnalysisFiles' 'VideoCrop.npy')})

        return cloud_files










