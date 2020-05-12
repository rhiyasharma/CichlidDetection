import argparse, os, subprocess, pdb, matplotlib, sys
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import os, tarfile
from PIL import Image
from matplotlib import rcParams
import numpy as np
import matplotlib.image as mpimg
import cv2

# 'command+Q' to shut the window; left arrow to display all modified image frames windows
# Run the script using the command: python3 prepareTrainingData.py <ProjectID>
# ex: python3 prepareTrainingData.py MC6_5
# 'CorrectAnnotations.csv' will be created that contains all the annotations that fall within the video crop
# Add '-v' or '--view' flag to view modified dataset


def initialising():
	rcloneRemote = 'd'
	output = subprocess.run(['rclone', 'lsf', rcloneRemote + ':'], capture_output = True, encoding = 'utf-8')
	if 'McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':McGrath/Apps/CichlidPiData/'
	elif 'BioSci-McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':BioSci-McGrath/Apps/CichlidPiData/'
	else:
		raise Exception('Cant find master McGrath directory in rclone remote')

	annotated_data_path = '__AnnotatedData/BoxedFish/BoxedFish.csv'

	image_data_path = '__AnnotatedData/BoxedFish/BoxedImages/{}'.format(imgfile_name) # Add .format{}; include argument to accept project ID

	remoteDir = cloudMasterDir + '{}'.format(args.ProjectID)
	remote_files = subprocess.run(['rclone', 'lsf', remoteDir], capture_output = True, encoding = 'utf-8')

	if 'videoCropPoints.npy' and 'videoCrop.npy' in remote_files.stdout.split():
		video_points_data_path = '{}/videoCropPoints.npy'.format(args.ProjectID)
		video_crop_data_path = '{}/videoCrop.npy'.format(args.ProjectID)
	else:
		video_points_data_path = '{}/MasterAnalysisFiles/VideoPoints.npy'.format(args.ProjectID)
		video_crop_data_path = '{}/MasterAnalysisFiles/VideoCrop.npy'.format(args.ProjectID)

	localDir = os.getenv('HOME') + '/' + 'Desktop/McGrathLab/'
	if not os.path.exists(localDir):
		os.mkdir(localDir)

	return cloudMasterDir, annotated_data_path, image_data_path, video_points_data_path, video_crop_data_path, localDir

# Download files from cloud directory to local directory and read it in
def download(cloudMasterDir, annotated_data, image_data, video_points_file, video_crop_file, localDir, imgfile_name):

	# Creating a subdirectory
	subdir_name = image_data.split('/')[3].split('.')[0]
	subdir = os.path.join(localDir, subdir_name)

	# Check if subdir is already there, and create it if it is not
	if not os.path.exists(subdir):
		os.mkdir(subdir)

	remoteDir = cloudMasterDir + '{}'.format(args.ProjectID)
	remote_files = subprocess.run(['rclone', 'lsf', remoteDir], capture_output = True, encoding = 'utf-8')


	os.chdir(subdir)
	localDir = os.getcwd()

	files = os.listdir()

	file1 = 'BoxedFish.csv'
	file2 = imgfile_name
	if 'videoCropPoints.npy' in remote_files.stdout.split():
		file3 = 'videoCropPoints.npy'
	else:
		file3 = 'VideoPoints.npy'
	file4 = 'VideoCrop.npy'


	if file1 not in files:
		print('Downloading annotations')
		# Download annotations
		downloaded_file_1 = subprocess.run(['rclone', 'copy', cloudMasterDir + annotated_data, localDir], stderr = subprocess.PIPE)

	if file2 not in files:
		print('Downloading images')
		# Download images
		downloaded_file_2 = subprocess.run(['rclone', 'copy', cloudMasterDir + image_data, localDir], stderr = subprocess.PIPE)

	if file3 not in files:
		print('Downloading video points')
		# Download video points file
		downloaded_file_3 = subprocess.run(['rclone', 'copy', cloudMasterDir + video_points_file, localDir], stderr = subprocess.PIPE)

	if file4 not in files:
		print('Downloading video crop')
		# Download video points file
		downloaded_file_4 = subprocess.run(['rclone', 'copy', cloudMasterDir + video_crop_file, localDir], stderr = subprocess.PIPE)


	return localDir, file1, file2, file3, file4

# --------------------------------- Area Calculation ---------------------------------#

def area(box):

	if 'VideoPoints.npy' in os.listdir():
		vpfile = 'VideoPoints.npy'
	elif 'videoCropPoints.npy' in os.listdir():
		vpfile = 'videoCropPoints.npy'

	# video points array:
	vp_array = np.load(vpfile)

	# Video crop coordinates
	v0 = list(vp_array[0])
	# v0_x, v0_y = v0
	v1 = list(vp_array[1])
	# v1_x, v0_y = v1
	v2 = list(vp_array[2])
	# v2_x, v0_y = v0
	v3 = list(vp_array[3])
	# v3_x, v0_y = v0


	# Create polygon for the video points 
	poly_vp = Polygon([v0, v1, v2, v3])

	# Annotation box coordinates
	x_a, y_a, w_a, h_a = box

	# Create polygon for the annotation box 
	poly_ann = Polygon([[x_a,y_a],[x_a+w_a,y_a],[x_a+w_a,y_a+h_a],[x_a,y_a+h_a]])

	intersection_area = poly_ann.intersection(poly_vp).area 
	ann_area = poly_ann.area

	return (intersection_area, ann_area)

# --------------------------------- Determining Whether Annotations Lie Within Crop ---------------------------------#

def determine(area):
	intersection_area, ann_area = area
	if intersection_area == ann_area:
		verdict = "Yes"
	else:
		verdict = "No"

	return verdict

# --------------------------------- Plot Images ---------------------------------#
'''
def plotImage(data):

	# video points array:
	vp_array = np.load('VideoPoints.npy')

	# Video crop coordinates
	v0 = list(vp_array[0])
	v1 = list(vp_array[1])
	v2 = list(vp_array[2])
	v3 = list(vp_array[3])

	x_vp = v0[0]
	y_vp = v0[1]
	# height calculation
	x2_vp = v1[0]
	h_vp = x2_vp - x_vp
	# width calculation
	y2_vp = v3[1]
	w_vp = y2_vp - y_vp

	frame = data[0]
	box = data[1]
	x_a, y_a, w_a, h_a = box

	img_file = 'MC_fem_con1.tar'
	if img_file in os.listdir():
		tf = tarfile.open(img_file)
		tf.extractall()

	img_dir = img_file.split('.')[0]	
	img = mpimg.imread(img_dir+'/'+frame)
	plt.imshow(img)
	ax = plt.gca()
	ax.add_patch(matplotlib.patches.Rectangle((x_a, y_a), w_a, h_a, linewidth=1, edgecolor='blue', facecolor='none'))
	ax.add_patch(matplotlib.patches.Rectangle((x_vp, y_vp), w_vp, h_vp, linewidth=1, edgecolor='red', facecolor='none'))

	plt.title(frame)
	plt.show()
'''


def prep_data(pid, view=False):
	# --------------------------------- Downloading The Data ---------------------------------#
	imgfile_name = '{}.tar'.format(pid)
	cloud_master_directory, annotated_data_path, image_file_path, video_points_file_path, video_crop_file_path, local_directory = initialising()
	localDir, ann_file, img_file, vp_file, vc_file = download(cloud_master_directory, annotated_data_path, image_file_path, video_points_file_path, video_crop_file_path, local_directory, imgfile_name)
	print("Download complete for {}!".format(pid))

	# --------------------------------- Iterating Through Annotations ---------------------------------#

	# video points array:
	vp_array = np.load(vp_file)

	# Video crop coordinates
	x_vp = tuple(vp_array[0])
	y_vp = tuple(vp_array[1])
	w_vp = tuple(vp_array[2])
	h_vp = tuple(vp_array[3])

	# Read in BoxFile.csv file
	df = pd.read_csv(localDir + '/' + ann_file)
	df = df[df['ProjectID'] == pid]
	df = df[df['CorrectAnnotation'] == 'Yes']
	df = df.fillna('')
	df = df[df['Box'] != '']
	df['Box']=[eval(i) for i in df['Box']]
	pd.set_option('display.max_rows', None)
	df['Area'] = df['Box'].apply(area)

	#  Iterate through the annotations and identify annotations that fall within the video crop
	df['WithinCrop'] = df['Area'].apply(determine)
	df = df[df['WithinCrop'] == 'Yes']
	# print(df)
	export_CorrectAnn_csv = df.to_csv(r'CorrectAnnotations.csv', index = None, header=True)
	framefiles = list(df.groupby('Framefile').count().index)
	print('Number of annotated frames that lie within the video crop: ', len(framefiles))
	if view:
		for frame in framefiles:
			#df_row = df[df['Framefile'] == frame]
			#df_row[['Framefile', 'Box']].apply(plotImage, axis=1)
			img_dir = img_file.split('.')[0]
			if img_dir not in os.listdir():
				tf = tarfile.open(img_file)
				tf.extractall()
			image_frame = img_dir+'/'+frame
			img = cv2.imread(image_frame)
			mask = np.load(vc_file)
			mask = np.logical_not(mask)
			img[mask] = 0
			cv2.imshow("Modified Frame: " + frame, img)
			cv2.waitKey(0)








