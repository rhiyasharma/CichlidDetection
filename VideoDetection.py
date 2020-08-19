import os
import cv2
import time
import argparse
import pandas as pd
from time import ctime
from CichlidDetection.Classes.Detector import Detector
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.FileManager import ProjectFileManager
from CichlidDetection.Classes.VideoCreator import Animation

# from CichlidDetection.Classes.DetectionsAnalysis import DetectionsAnalysis

# parse command line arguments
parser = argparse.ArgumentParser(description='To Detect Cichlids in Videos')
parser.add_argument('pid', type=str, metavar=' ', help='Project ID. Ex: MC6_5')
parser.add_argument('video', type=str, metavar=' ', help='Run detection on specified video. Ex: 0005_vid.mp4')
parser.add_argument('-i', '--download_images', action='store_true', help='Download full image directory')
parser.add_argument('-v', '--download_video', action='store_true', help='Download video')
args = parser.parse_args()

"""
Download videos from different projects and run them through the model to detect cichlids

Args:
    pid (str): project id
    download_images (bool): if True, download the full image directory for the specified project
    download_videos (bool): if True, download the all the mp4 files in Videos directory for the specified project
    video (str): specifies which video to download
"""


def calc_video_intervals(video):
    # Create a list of intervals for the model
    cap = cv2.VideoCapture(video)
    intervals = []
    len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # limit for the number of frames that can be loaded at once: 46000
    for x in range(len):
        if x % 23000 == 0:
            intervals.append(x)

    if len % 23000 != 0:
        intervals.append(len)

    cap.release()
    return intervals


s = ctime(time.time())
print("Start Time (Full): ", ctime(time.time()))

fm = FileManager()
# Create project directory and download the specified files
pfm = ProjectFileManager(args.pid, fm, args.download_images, args.download_video, args.video)
print('downloaded video, created directories!')

print("Start Detect Time: ", ctime(time.time()))
detect = Detector(pfm)
# # video_path = os.path.join('/Users/rhiyasharma/Documents/_McGrathLab/CD_work/videos', args.video)
video_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], args.video)
video_name = args.video.split('.')[0]

# Calculate intervals for the videos
intervals_list = calc_video_intervals(video_path)
for i in range(len(intervals_list)):
    start = intervals_list[i]
    end = intervals_list[i + 1]
    detect.frame_detect(args.pid, video_path, start, end)

print("End Detect Time: ", ctime(time.time()))

print('Creating the final consolidated csv file')
identifying_name = '{}_{}'.format(args.pid, video_name)
csv_files = []
for file in os.listdir(fm.local_files['detect_dir']):
    if identifying_name in file:
        df = pd.read_csv(file, index_col='Framefile', header=0)
        csv_files.append(df)

csv_final = pd.concat(csv_files)
csv_final.to_csv('{}_detections_final.csv'.format(identifying_name))
print('Created {}_detections_final.csv'.format(identifying_name))

print('Starting the animation process...')
animation = Animation(args.pid, args.video, csv_final, pfm)
animation.animated_learning()
print('Detections video made!')

print("Start Time (Full): ", s)
print("End Time (Full): ", ctime(time.time()))

# csv_file_name = '/Users/rhiyasharma/Documents/_McGrathLab/CD_work/csv/detections_new_Ordered.csv'
# analysis = DetectionsAnalysis(csv_file_name, pfm)
