import os
from CichlidDetection.Classes.DataPrepper import DataPrepper
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.Trainer import Trainer
from CichlidDetection.Classes.Detector import Detector


class Runner:
    """user-friendly class for accessing the majority of module's functionality."""
    def __init__(self):
        """initiate the Runner class"""
        self.fm = FileManager()
        self.dp = DataPrepper()
        self.tr = None
        self.de = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def download(self):
        """download all required data."""
        self.dp.download_all()

    def prep(self):
        """prep downloaded data"""
        self.dp.prep()

    def train(self, num_epochs, upload_results=True):
        """initiate a Trainer object and train the model.

        Args:
            num_epochs (int): number of epochs to train
            upload_results(bool): if True, automatically upload the results (weights, logs, etc.) after training
        """
        self.tr = Trainer(num_epochs, upload_results)
        self.tr.train()

    def sync(self):
        self.fm.sync_training_dir()

    def detect(self, img_dir):
        self.de = Detector()
        if img_dir == 'test':
            self.de.test(10)
        else:
            self.de.detect(img_dir)
