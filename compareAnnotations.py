import pdb, matplotlib, os, argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
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
		gender_actual = []
		gender_pred=[]

		for row in dt.itertuples():
			ann1 = row.Box_man
			ann2 = row.Box_ml
			sex1 = row.Sex_man
			sex2 = row.Sex_ml
			iou = row.IOU
			actual_sex = eval(row.Sex_man)
			pred_sex = eval(row.Sex_ml)

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


			if len(set(pred_sex)) == 0:
				gender_pred.append(0)
			elif len(set(pred_sex)) == 1:
				if pred_sex[0] == 1:
					gender_pred.append(1)
				else:
					gender_pred.append(2)
			elif len(set(pred_sex)) == 2:
				gender_pred.append(3)

		dt['SexAgree'] = pd.Series(sex_agree)
		# dt['Gender_Actual'] = pd.Series(gender_actual)
		dt['Gender_Predicted'] = pd.Series(gender_pred)

		dt = dt[['Framefile', 'Box_man', 'Sex_man', 'Box_ml', 'Sex_ml', 'Gender_Predicted', 'SexAgree', 'IOU', 'avg_accuracy']]
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
parser.add_argument('-v', '--view', action = 'store_true', help = 'Use this flag to view the disagreements')
parser.add_argument('-s', '--save', action='store_true', help='Use this flag to save dataframe as csv')
parser.add_argument('-p', '--plot', action='store_true', help='Use this flag to plot male/female analysis')
args = parser.parse_args()

comparer = CompareAnnotations()
dt1 = pd.read_csv(comparer.data)
dt = CompareAnnotations.compare(dt1)
dt_f = dt[dt['Gender_Predicted'] == 1]
dt_m = dt[dt['Gender_Predicted'] == 2]
dt_b = dt[dt['Gender_Predicted'] == 3]
dt_n = dt[dt['Gender_Predicted'] == 0] # No fish; 82 frames with no fish

dt_f = dt_f.groupby(dt_f['avg_accuracy']).count().reset_index() # females
dt_m = dt_m.groupby(dt_m['avg_accuracy']).count().reset_index() # males
dt_b = dt_b.groupby(dt_b['avg_accuracy']).count().reset_index() # both


if args.save:
	dt.to_csv(join(comparer.csv_dir, 'CompareAnnotations.csv'))

if args.view:
	dt_disagreements = dt[dt.SexAgree == 'False']
	framefiles = dt_disagreements.groupby('Framefile').count().index
	for frame in framefiles:
		CompareAnnotations.plotPhoto(frame, dt_disagreements, comparer.img_dir)



##### Plotting #####
fig = plt.figure()
# fig.set_size_inches(11, 8.5)
# fig.tight_layout()
ax1 = fig.add_subplot(221)   #top left
ax2 = fig.add_subplot(222)   #top right
ax3 = fig.add_subplot(223)   #bottom left
ax4 = fig.add_subplot(224)   #bottom right

#### Plot1
colors=['lightgray', 'lightpink', 'royalblue', 'red']
sns.barplot(x="Gender_Predicted", y="avg_accuracy", hue="Gender_Predicted", data=dt, order=[1,2,3,0], dodge=False, ax=ax1, palette=colors)
ax1.set_ylabel('Average Accuracy', fontsize=7)
ax1.set_xlabel('Genders', fontsize=7)
ax1.set_xticklabels(['Female', 'Male', 'Both', 'No Fish'])
ax1.set_title('General Analysis', fontsize=10)
ax1.get_legend().set_visible(False)


#### Plot 2
x2_labels = dt_f.avg_accuracy.round(2)
sns.barplot(x="avg_accuracy", y="Framefile", data=dt_f, color='lightpink', ax=ax2)
ax2.set_xlabel('Average Accuracy', fontsize=7)
ax2.set_xticklabels(x2_labels)
ax2.set_ylabel('No. of Framefiles', fontsize=7)
ax2.set_title('Only Females Present', fontsize=10)


#### Plot 3
x3_labels = dt_m.avg_accuracy.round(2)
sns.barplot(x="avg_accuracy", y="Framefile", data=dt_m, color='royalblue', ax=ax3)
ax3.set_xlabel('Average Accuracy', fontsize=7)
ax3.set_xticklabels(x3_labels)
ax3.set_ylabel('No. of Framefiles', fontsize=7)
ax3.set_title('Only Males Present', fontsize=10)


#### Plot 4
x4_labels = dt_b.avg_accuracy.round(2)
sns.barplot(x="avg_accuracy", y="Framefile", data=dt_b, color='orange', ax=ax4)
ax4.set_xlabel('Average Accuracy', fontsize=7)
ax4.set_xticklabels(x4_labels)
ax4.set_ylabel('No. of Framefiles', fontsize=7)
ax4.set_title('Both Males & Females Present', fontsize=10)

plt.show()

