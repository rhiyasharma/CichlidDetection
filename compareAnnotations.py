import pdb
import pandas as pd
import numpy as np
import os
from CichlidDetection.Classes.FileManager import FileManager

class CompareAnnotations:

	def __init__(self):
		self.fm = FileManager()
		self.csv_dir = self.fm.local_files['figure_data_dir']
		self.file='epoch_99_eval.csv'
		self.data = os.path.join(self.csv_dir, self.file)

	def compare(dt):

		dt.rename(columns={'boxes_actual': 'Box_man', 'labels_actual': 'Sex_man', 'boxes_predicted': 'Box_ml',
						 'labels_predicted': 'Sex_ml', 'average_iou': 'IOU'}, inplace=True)

		sex_agree = []

		for row in dt.itertuples():
			ann1 = row.Box_man
			ann2 = row.Box_ml
			sex1 = row.Sex_man
			sex2 = row.Sex_ml
			iou = row.IOU

			try:
				ann1 = eval(ann1)
				ann2 = eval(ann2)
			except TypeError:
				sex_agree.append(np.nan)
				continue


			if iou != float(1.0):
				if sex1 == sex2:
					sex_agree.append('True')
				else:
					sex_agree.append('False')
			else:
				sex_agree.append('No fish')

		dt['SexAgree'] = pd.Series(sex_agree)

		dt = dt[['Box_man', 'Sex_man', 'Box_ml', 'Sex_ml', 'SexAgree', 'IOU']]
		dt.to_csv('new_compare_ann.csv')


comparer = CompareAnnotations()
dt1 = pd.read_csv(comparer.data)
CompareAnnotations.compare(dt1)
