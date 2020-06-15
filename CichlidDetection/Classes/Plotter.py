from CichlidDetection.Classes.FileManagers import FileManager
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
from matplotlib.figure import Figure


def plotter_decorator(plotter_method):
    """decorator used for automatic set-up and clean-up when making figures with methods from the Plotter class"""
    @wraps(plotter_method)
    def wrapper(plotter, fig=None, *args, **kwargs):
        method_name = plotter_method.__name__
        fig = plt.Figure(*args, **kwargs) if fig is None else fig
        plotter_method(plotter, fig)
        plotter.save_fig(fig, method_name)
    return wrapper


class Plotter:

    def __init__(self):
        self.fm = FileManager()
        self.fig_dir = self.fm.local_files['figure_dir']
        self._load_data()

    def save_fig(self, fig: Figure, file_stub: str):
        """save the figure as a pdf and close it

        Notes:
            saves the figure to the figure_dir specified in the FileManager object

        Args:
            fig (Figure): figure to save
            file_stub (str): name to use for the file. Don't include '.pdf'
        """
        fig.savefig(join(self.fig_dir, '{}.pdf'.format(file_stub)))
        plt.close('all')

    def plot_all(self):
        pass

    @plotter_decorator
    def loss_vs_epoch(self, fig: Figure):
        """plot the training loss vs epoch and save as loss_vs_epoch.pdf

        Args:
            fig (Figure): matplotlib Figure object into which to plot
        """
        ax = fig.add_subplot(111)
        ax.set(xlabel='epoch', ylabel='loss', title='Training Loss vs. Epoch')
        sns.lineplot(data=self.train_log.loss)

    def _load_data(self):
        """load and parse all relevant data"""
        self.train_log = self._parse_train_log()
        self.num_epochs = len(self.train_log)
        self.ground_truth = self._parse_epoch_csv()
        self.epoch_predictions = []
        for epoch in range(self.num_epochs):
            self.epoch_predictions.append(self._parse_epoch_csv(epoch))

    def _analyze_data(self):
        """calculate relevant summary stats, metrics, etc. from the raw data"""

    def _parse_train_log(self):
        """parse the logfile that tracked overall loss and learning rate at each epoch

        Returns:
            Pandas Dataframe, indexed by epoch number, with the columsn 'loss' and 'lr'
        """
        return pd.read_csv(self.fm.local_files['train_log'], sep='\t', index_col='epoch')

    def _parse_epoch_csv(self, epoch=-1):
        """parse the csv file of predictions produced when Trainer.train() is run with compare_annotations=True

        Notes:
            if the epoch arg is left at the default value of -1, this function will instead parse 'ground_truth.csv'

        Args:
            epoch(int): epoch number, where 0 refers to the first epoch. Defaults to -1, which parses the
                ground truth csv

        Returns:
            Pandas DataFrame of epoch data
        """
        if epoch == -1:
            return pd.read_csv(self.fm.local_files['ground_truth_csv'])
        else:
            return pd.read_csv(join(self.fm.local_files['predictions_dir'], '{}.csv'.format(epoch)))
