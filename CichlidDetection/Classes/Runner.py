import argparse, os, sys, subprocess
from CichlidDetection.Classes.DataPreppers import DataPrepper
from CichlidDetection.Classes.FileManagers import FileManager
from CichlidDetection.Utilities.system_utilities import run


class Runner:
    def __init__(self):
        self.fm = FileManager()
        self.dp = DataPrepper()
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        pass

    def download(self):
        self.dp.download_all()

    def prep(self):
        self.dp.prep()

    def train(self):
        pass

    def test(self):
        pass
