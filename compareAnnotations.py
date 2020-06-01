import pdb
import pandas as pd
import numpy as np

def compareAnnotations(man_dt, ml_dt, joining_columns_man = ['ProjectID','Framefile'], joining_columns_ml = ['ProjectID','Framefile'], annotation_man = 'Box', annotation_ml = 'Box', sex_man = 'Sex', sex_ml = 'Sex'):
	man_dt.rename(columns={annotation_man: 'Box_man', sex_man: 'Sex_man'}, inplace=True)
	ml_dt.rename(columns={annotation_ml: 'Box_ml', sex_ml: 'Sex_ml'}, inplace=True)

	dt = pd.merge(man_dt, ml_dt, left_on = joining_columns_man, right_on = joining_columns_man, how = 'left')

	# Add IOU
	ious = []
	sex_agree = []
	overlap = []

	for row in dt.itertuples():
		ann1 = row.Box_man
		ann2 = row.Box_ml
		sex1 = row.Sex_man
		sex2 = row.Sex_ml

		try:
			ann1 = eval(ann1)
			ann2 = eval(ann2)
		except TypeError:
			ious.append(np.nan)
			sex_agree.append(np.nan)
			overlap.append('False')
			continue

		overlap_x0, overlap_y0, overlap_x1, overlap_y1 = max(ann1[0],ann2[0]), max(ann1[1],ann2[1]), min(ann1[0] + ann1[2],ann2[0] + ann2[2]), min(ann1[1] + ann1[3],ann2[1] + ann2[3])
		if overlap_x1 < overlap_x0 or overlap_y1 < overlap_y0:
			ious.append(0)
			sex_agree.append(np.nan)
			overlap.append('False')
		else:
			intersection = (overlap_x1 - overlap_x0)*(overlap_y1 - overlap_y0)
			union = ann1[2]*ann1[3] + ann2[2]*ann2[3] - intersection
			ious.append(intersection/union)
			if sex1 == sex2:
				sex_agree.append('True')
			else:
				sex_agree.append('False')
			overlap.append('True')


	dt['IOU'] = pd.Series(ious)
	dt['Overlap'] = pd.Series(overlap)
	dt['SexAgree'] = pd.Series(sex_agree)

	idx = dt.groupby(joining_columns_man + ['Box_man'])['IOU'].transform(max) == dt['IOU']
	dt = dt[idx].reset_index()
	
dt1 = pd.read_csv('/Users/pmcgrath7/Dropbox (GaTech)/McGrath/Apps/CichlidPiData/__AnnotatedData/BoxedFish/BoxedFish.csv')
dt2 = pd.read_csv('/Users/pmcgrath7/Dropbox (GaTech)/McGrath/Apps/CichlidPiData/__AnnotatedData/BoxedFish/BoxedFish.csv')

compareAnnotations(dt1,dt2)