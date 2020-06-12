import os
from CichlidDetection.Classes.DataPreppers import DataPrepper
from CichlidDetection.Classes.FileManagers import FileManager
from CichlidDetection.Classes.Trainers import Trainer


class Runner:
    """user-friendly class for accessing the majority of module's functionality."""
    def __init__(self):
        """initiate the Runner class"""
        self.fm = FileManager()
        self.dp = DataPrepper()
        self.tr = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def download(self):
        """download all required data."""
        self.dp.download_all()

    def prep(self):
        """prep downloaded data"""
        self.dp.prep()

    def train(self, num_epochs):
        """initiate a Trainer object and train the model.

        Args:
            num_epochs (int): number of epochs to train
        """
        self.tr = Trainer(num_epochs)
        self.tr.train()
