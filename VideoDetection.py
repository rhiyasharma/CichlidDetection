import os, subprocess
import cv2
import time
from os.path import join
import argparse
import pandas as pd
from time import ctime
from itertools import chain
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
parser.add_argument('-s', '--sync', action='store_true', help='Sync detections directory')
args = parser.parse_args()

"""
Download videos from different projects and run them through the model to detect cichlids

Args:
    pid (str): project id
    download_images (bool): if True, download the full image directory for the specified project
    download_videos (bool): if True, download the all the mp4 files in Videos directory for the specified project
    video (str): specifies which video to download
    sync (bool): if True, upload the final csv and animation video to the cloud
"""


def calcIntervals(video):
    # Create a list of intervals for the model
    cap = cv2.VideoCapture(video)
    intervals = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # limit for the number of frames that can be loaded at once: 18000
    if length > 18000:
        vid_dir_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], video_name)
        if not os.path.exists(vid_dir_path):
            pfm.local_files.update({video_name: make_dir(vid_dir_path)})
        else:
            pfm.local_files.update({video_name: vid_dir_path})

    for x in range(length):
        if x % 18000 == 0:
            intervals.append(x)

    if length % 18000 != 0:
        intervals.append(length)

    cap.release()
    return intervals


def clipVideos(video, name, begin, end, frame_num):
    # Create short 10 min clips of the 6-10h long video
    # Open video & get video details
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    # Set video name, start & stop intervals
    vid_name = name + '_{}.mp4'.format(begin)

    # Output video details
    result = cv2.VideoWriter(os.path.join(pfm.local_files[name], '{}'.format(vid_name)),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

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
    return location, frame_num


def sync_detection_dir(exclude=None, quiet=False):
    """sync the detection directory bidirectionally, keeping the newer version of each file

            Args:
                exclude (list of str): files/directories to exclude. Accepts both explicit file/directory names and
                    regular expressions. Expects a list, even if it's a list of length one. Default None.
                quiet: if True, suppress the output of rclone copy. Default False
            """
    print('syncing training directory')
    cloud_detection_dir = join(fm.cloud_master_dir, '___Tucker', 'CichlidDetection', 'detection')
    down = ['rclone', 'copy', '-u', '-c', cloud_detection_dir, fm.local_files['detection_dir']]
    up = ['rclone', 'copy', '-u', '-c', fm.local_files['detection_dir'], cloud_detection_dir, '--exclude',
          '.*{/**,}']
    if not quiet:
        [com.insert(3, '-P') for com in [down, up]]
    if exclude is not None:
        [com.extend(list(chain.from_iterable(zip(['--exclude'] * len(exclude), exclude)))) for com in [down, up]]
    [run(com) for com in [down, up]]


########################################################## Program starts here ##########################################################


# Measure duration of program
s = ctime(time.time())
print("Start Time (Full): ", ctime(time.time()))

# Initialize functions
fm = FileManager()
# Create project directory and download the specified files
pfm = ProjectFileManager(args.pid, fm, args.download_images, args.download_video, args.video)
print('downloaded video, created directories!')

video_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], args.video)
video_name = args.video.split('.')[0]

# Create intervals list and iterate through them to crop video and feed it into the model
if 'sample' in video_name:
    detect = Detector(pfm)
    print("Start Detect Time: ", ctime(time.time()))
    detect.frame_detect(args.pid, video_path)
    print("End Detect Time: ", ctime(time.time()))
else:
    detect = Detector(pfm)
    interval_list = calcIntervals(video_path)
    video_list=[]
    count = 0
    for i in range(len(interval_list)-1):
        print('Starting video {}...'.format(i))
        start = interval_list[i]
        stop = interval_list[i+1]
        vid_location, num = clipVideos(video_path, video_name, start, stop, count)
        count = num
        video_list.append(vid_location)
        print('Attempting detection for video {}'.format(i))
        print("Start Detect Time: ", ctime(time.time()))
        detect.frame_detect(args.pid, vid_location)
        print("End Detect Time: ", ctime(time.time()))

    print('{} was successively split into {} parts'.format(video_name, len(video_list)))

csv_list = os.listdir(pfm.local_files['detection_dir'])
csv_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
df_list = []
for i in csv_list:
    csv_path = os.path.join(pfm.local_files['detection_dir'], i)
    df = pd.read_csv(csv_path)
    df_list.append(df)

final_csv = pd.concat(df_list, axis=0)
csv_name = '{}_{}_detections.csv'.format(args.pid, video_name)
csv_location = os.path.join(pfm.local_files['detection_dir'], csv_name)
final_csv.to_csv(csv_location)
print("Final csv: ", csv_name)

print('Deleting the other csv files...')
for i in csv_list:
    if i != csv_name:
        csv_path = os.path.join(pfm.local_files['detection_dir'], i)
        subprocess.run(['rm', csv_path])

print('Deleting {} clipped videos...'.format(args.video))
subprocess.run(['rm', '-rf', pfm.local_files[video_name]])

csv_name = '{}_{}_detections.csv'.format(args.pid, video_name)
print('Starting the video annotation process...')
video_ann = VideoAnnotation(args.pid, video_path, args.video, csv_name, pfm)
video_ann.annotate()

print('Process complete!')

if args.sync:
    sync_detection_dir()

print("Start Time (Full): ", s)
print("End Time (Full): ", ctime(time.time()))
