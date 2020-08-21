import os
# import cv2
import time
import argparse
# import pandas as pd
from time import ctime
from CichlidDetection.Classes.Detector import Detector
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.FileManager import ProjectFileManager
from CichlidDetection.Classes.VideoCreator import VideoAnnotation

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


# def calc_video_intervals(video):
#     # Create a list of intervals for the model
#     cap = cv2.VideoCapture(video)
#     intervals = []
#     len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # limit for the number of frames that can be loaded at once: 46000
#     for x in range(len):
#         if x % 23000 == 0:
#             intervals.append(x)
#
#     if len % 23000 != 0:
#         intervals.append(len)
#
#     cap.release()
#     return intervals


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

detect.frame_detect(args.pid, video_path)
print("End Detect Time: ", ctime(time.time()))

csv_file_name = '{}_{}_detections.csv'.format(args.pid, video_name)

print('Starting the video annotation process...')
video_ann = VideoAnnotation(args.pid, args.video, csv_file_name, pfm)
video_ann.annotate()

print('Process complete!')

print("Start Time (Full): ", s)
print("End Time (Full): ", ctime(time.time()))

# csv_file_name = '/Users/rhiyasharma/Documents/_McGrathLab/CD_work/csv/detections_new_Ordered.csv'
# analysis = DetectionsAnalysis(csv_file_name, pfm)
