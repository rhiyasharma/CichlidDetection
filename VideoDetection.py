import os
import cv2
import time
import argparse
# import pandas as pd
from time import ctime
from CichlidDetection.Classes.Detector import Detector
from CichlidDetection.Utilities.utils import run, make_dir
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.VideoCreator import VideoAnnotation
from CichlidDetection.Classes.FileManager import ProjectFileManager

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


def calcIntervals(video):
    # Create a list of intervals for the model
    cap = cv2.VideoCapture(video)
    intervals = []
    len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # limit for the number of frames that can be loaded at once: 18000
    if len > 18000:
        vid_dir_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], video_name)
        if not os.path.exists(vid_dir_path):
            pfm.local_files.update({video_name: make_dir(vid_dir_path)})

    for x in range(len):
        if x % 18000 == 0:
            intervals.append(x)

    if len % 18000 != 0:
        intervals.append(len)

    cap.release()
    return intervals


def clipVideos(video, name, begin, end, frame_num):
    # Create short 10 min clips of the 6-10h long video
    # Open video & get video details
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    os.chdir(pfm.local_files[name])

    # Set video name, start & stop intervals
    vid_name = name + '_{}.mp4'.format(begin)

    # Output video details
    result = cv2.VideoWriter('{}'.format(vid_name), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

    for j in range(begin, end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ret, frame = cap.read()
        if not ret:
            print('Bad frame')
            break
        else:
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        frame_num += 1

    cap.release()
    result.release()
    cv2.destroyAllWindows()

    print("Video: {} was successfully saved".format(vid_name))
    location = os.path.join(pfm.local_files[name], vid_name)
    return location


########################################################## Program starts here ##########################################################


# Measure duration of program
s = ctime(time.time())
print("Start Time (Full): ", ctime(time.time()))

# Initialize functions
fm = FileManager()
# Create project directory and download the specified files
pfm = ProjectFileManager(args.pid, fm, args.download_images, args.download_video, args.video)
print('downloaded video, created directories!')

# # video_path = os.path.join('/Users/rhiyasharma/Documents/_McGrathLab/CD_work/videos', args.video)
video_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], args.video)
video_name = args.video.split('.')[0]

# Create intervals list and iterate through them to crop video and feed it into the model
print("Start Detect Time: ", ctime(time.time()))
detect = Detector(pfm)
interval_list = calcIntervals(video_path)
video_list=[]
count = 0
for i in range(len(interval_list)-1):
    start = interval_list[i]
    stop = interval_list[i+1]
    vid_location = clipVideos(video_path, video_name, start, stop, count)
    video_list.append(vid_location)
    detect.frame_detect(args.pid, vid_location)

print('{} was successively split into {} parts'.format(video_name, len(video_list)))

print("End Detect Time: ", ctime(time.time()))


# csv_file_name = '{}_{}_detections.csv'.format(args.pid, video_name)

# print('Starting the video annotation process...')
# video_ann = VideoAnnotation(args.pid, args.video, csv_file_name, pfm)
# video_ann.annotate()

print('Process complete!')

print("Start Time (Full): ", s)
print("End Time (Full): ", ctime(time.time()))

# csv_file_name = '/Users/rhiyasharma/Documents/_McGrathLab/CD_work/csv/detections_new_Ordered.csv'
# analysis = DetectionsAnalysis(csv_file_name, pfm)
