import argparse, os, sys
from CichlidDetection.Classes.DataPrepper import DataPrepper
from CichlidDetection.Classes.FileManager import FileManager

# main script


class Runner:
    def __init__(self):
        pass

    def prep(self, pid):
        fm = FileManager(pid)
        dp = DataPrepper(fm)
        dp.prep_annotations()
        dp.YOLO_prep()
        return fm

    def train(self):
        pass
