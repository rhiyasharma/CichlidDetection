import argparse, os, sys, subprocess
from CichlidDetection.Classes.DataPrepper import DataPrepper
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Utilities.SystemUtilities import run

# main script


class Runner:
    def __init__(self):
        self.fm = None
        self.dp = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        pass

    def prep(self, pid):
        self.fm = FileManager(pid)
        self.dp = DataPrepper(self.fm)
        self.dp.YOLO_prep()

    def train(self):
        if self.fm is None:
            print('No file manager detected. Run prep prior to training')
        else:
            model_dir = os.path.join(self.__location__, 'Classes', 'Models', 'YOLO')
            pbs_file = os.path.join(model_dir, 'train.pbs')
            data_file = self.fm.local_files['data_file']
            cmd = ['qsub', '-v', 'DATA_FILE={},MODEL_DIR={}'.format(data_file, model_dir), pbs_file]
            run(cmd)
            print('training initiated. Use qstat to check job status')
