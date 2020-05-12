import os, subprocess


def make_dir(path):
    """recursively create the directory specified by path if it does not exist
    :param path: path to the directory that will be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), "failed to create {}".format(path)


class FileManager:

    def __init__(self, pid):
        self.pid = pid
        self.project_dir = self.initialize_project_directory()

    def initialize_project_directory(self):
        project_dir = os.path.join(os.getenv('HOME'), 'scratch', 'CichlidDetection', self.pid)
        make_dir(project_dir)
        return project_dir

    def download(self, source, destination=None):
        destination = self.project_dir if destination is None else destination
        subprocess.run(['rclone', 'copy', source, destination])







