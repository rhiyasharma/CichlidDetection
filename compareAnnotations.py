import pdb, matplotlib, os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, exists
import matplotlib.image as mpimg
from CichlidDetection.Classes.FileManager import FileManager

class CompareAnnotations:

	def __init__(self):
		self.fm = FileManager()
		self.csv_dir = self.fm.local_files['figure_data_dir']
		self.file='epoch_99_eval.csv'
		self.data = os.path.join(self.csv_dir, self.file)
		self.img_dir = self.fm.local_files['test_image_dir']

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

		dt = dt[['Framefile', 'Box_man', 'Sex_man', 'Box_ml', 'Sex_ml', 'SexAgree', 'IOU']]
		return dt

	def plotPhoto(frame, dt2, img_dir):
		img = mpimg.imread(img_dir + '/' + frame)
		plt.imshow(img)
		ax = plt.gca()

		annotations = dt2[dt2.Framefile == frame]

		for row in annotations.itertuples():
			if row.Box_man != '[]':
				box = eval(row.Box_man)
				sex = eval(row.Sex_man)
				num = len(sex)
				for i in range(num):
					if sex[i]==1:
						ax.add_patch(matplotlib.patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1],
																  linewidth=1, edgecolor='pink', facecolor='none'))
					elif sex[i]==2:
						ax.add_patch(matplotlib.patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1],
																  linewidth=1, edgecolor='blue', facecolor='none'))

		for row in annotations.itertuples():
			if row.Box_ml != '[]':
				box = eval(row.Box_ml)
				sex = eval(row.Sex_ml)
				num = len(sex)
				for i in range(num):
					if sex[i]==1:
						ax.add_patch(matplotlib.patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1],
																  linewidth=1.5, edgecolor='pink', facecolor='none', linestyle='--'))
					elif sex[i]==2:
						ax.add_patch(matplotlib.patches.Rectangle((box[i][0], box[i][1]), box[i][2] - box[i][0], box[i][3] - box[i][1],
																  linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--'))


		plt.title(frame)
		plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('-pl', '--plot', action = 'store_true', help = 'Use this flag to view the disagreements')
parser.add_argument('-s', '--save', action='store_true', help='Use this flag to save dataframe as csv')
args = parser.parse_args()

comparer = CompareAnnotations()
dt1 = pd.read_csv(comparer.data)
dt = CompareAnnotations.compare(dt1)

if args.save:
	dt.to_csv(join(comparer.csv_dir, 'CompareAnnotations.csv'))

dt_disagreements = dt[dt.SexAgree=='False']

framefiles = dt_disagreements.groupby('Framefile').count().index
if args.plot:
	for frame in framefiles:
		CompareAnnotations.plotPhoto(frame, dt_disagreements, comparer.img_dir)