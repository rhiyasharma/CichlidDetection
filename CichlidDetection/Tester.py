from .Classes.DataPreppers import DataPrepper
from .Classes.FileManagers import FileManager

trial = 'MC6_5'
fm = FileManager()
dp = DataPrepper()
dp.download_all()
dp.prep()

