from CichlidDetection.Classes.DataPrepper import DataPrepper
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.Trainer import Trainer
from CichlidDetection.Classes.Plotter import Plotter

DataPrepper()._generate_ground_truth_csv()
plotter = Plotter()
plotter.plot_all()

