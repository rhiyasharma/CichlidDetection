import os
from CichlidDetection.Classes.DataPreppers import DataPrepper
from CichlidDetection.Classes.FileManagers import FileManager
from CichlidDetection.Classes.Trainers import Trainer


class Runner:
    def __init__(self):
        self.fm = FileManager()
        self.dp = DataPrepper()
        self.tr = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def download(self):
        self.dp.download_all()

    def prep(self):
        self.dp.prep()

    def train(self, num_epochs):
        self.tr = Trainer(num_epochs)
        self.tr.train()

    def test(self):
        pass
