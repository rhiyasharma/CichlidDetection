# Example usage: python3 VideoDetection.py 'MC6_5' '0001_vid.mp4' -v -f -s

import pandas as pd
from time import ctime
from os.path import join
from itertools import chain
import os, subprocess, cv2, time, argparse
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
parser.add_argument('-f', '--full', action='store_true', help='Run complete program')
parser.add_argument('-a', '--annotate', action='store_true', help='Annotate video')
parser.add_argument('-s', '--sync', action='store_true', help='Sync detections directory')
args = parser.parse_args()

"""
Download videos from different projects and run them through the model to detect cichlids

Args:
    pid (str): project id
    download_images (bool): if True, download the full image directory for the specified project
    download_videos (bool): if True, download the all the mp4 files in Videos directory for the specified project
    video (str): specifies which video to download
    full (bool): if True, run all the processes - video trimming, detections, 
    sync (bool): if True, upload the final csv and annotated video to the cloud


    ~10h video files are too big to be processed on the server. Using calcIntervals() and clipVideos() to trim the 
    video into manageable chunks 

"""


def calcIntervals(video):
    """ Create a list of intervals for the video to be trimmed into

    Args:
            video (str): Path of the original video file
    """

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
    """ Create short 10 min clips of the 6-10h long video

        Args:
                video (str): Path of the original video file
                name (str): Name of the cropped video file
                begin (int): Starting frame number of the cropped video
                end (int): Final frame number of the cropped video
                frame_num (int): Tracking the frame numbers
        """

    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    vid_name = name + '_{}.mp4'.format(begin)

    # Trimmed video details
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
    """ Sync the detection directory bidirectionally, keeping the newer version of each file

        Args:
            exclude (list of str): files/directories to exclude. Accepts both explicit file/directory names and
            regular expressions. Expects a list, even if it's a list of length one. Default None.
            quiet: if True, suppress the output of rclone copy. Default False
    """

    print('syncing training directory')
    cloud_detection_dir = join(fm.cloud_master_dir, '___Rhiya', 'CichlidDetection', 'detection')
    down = ['rclone', 'copy', '-u', '-c', cloud_detection_dir, fm.local_files['detection_dir']]
    up = ['rclone', 'copy', '-u', '-c', fm.local_files['detection_dir'], cloud_detection_dir, '--exclude',
          '.*{/**,}']
    if not quiet:
        [com.insert(3, '-P') for com in [down, up]]
    if exclude is not None:
        [com.extend(list(chain.from_iterable(zip(['--exclude'] * len(exclude), exclude)))) for com in [down, up]]
    [run(com) for com in [down, up]]


# Measure duration of program
s = ctime(time.time())
print("Start Time (Full): ", ctime(time.time()))

# Initialize functions. Create project directory and download the specified files
fm = FileManager()
pfm = ProjectFileManager(args.pid, fm, args.download_images, args.download_video, args.video)
print('downloaded video, created directories!')

# Storing video path. Setting video and final csv file names.
video_path = os.path.join(pfm.local_files['{}_dir'.format(args.pid)], args.video)
video_name = args.video.split('.')[0]
csv_name = '{}_{}_detections.csv'.format(args.pid, video_name)

if args.full:
    """
        1. Run all the processes - video trimming, detections, video annotation
        2. Create intervals list and iterate through them to crop video and feed it into the model
    """

    if 'sample' in video_name:
        detect = Detector(pfm)
        print("Start Detect Time: ", ctime(time.time()))
        detect.frame_detect(args.pid, video_path)
        print("End Detect Time: ", ctime(time.time()))
    else:
        detect = Detector(pfm)
        interval_list = calcIntervals(video_path)
        video_list = []
        count = 0
        for i in range(len(interval_list) - 1):
            print('Starting video {}...'.format(i))
            start = interval_list[i]
            stop = interval_list[i + 1]
            vid_location, num = clipVideos(video_path, video_name, start, stop, count)
            count = num
            video_list.append(vid_location)
            print('Attempting detection for video {}'.format(i))
            print("Start Detect Time: ", ctime(time.time()))
            detect.frame_detect(args.pid, vid_location)
            print("End Detect Time: ", ctime(time.time()))

        print('{} was successively split into {} parts'.format(video_name, len(video_list)))

    # Create a consolidated detections csv file
    csv_list = os.listdir(pfm.local_files['detection_dir'])
    csv_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    df_list = []
    for i in csv_list:
        csv_path = os.path.join(pfm.local_files['detection_dir'], i)
        df = pd.read_csv(csv_path)
        df_list.append(df)

    final_csv = pd.concat(df_list, axis=0)
    csv_location = os.path.join(pfm.local_files['detection_dir'], csv_name)
    final_csv.to_csv(csv_location)
    print("Final csv: ", csv_name)

    # Getting rid of unnecessary csv
    print('Deleting the other csv files...')
    for i in csv_list:
        if i != csv_name:
            csv_path = os.path.join(pfm.local_files['detection_dir'], i)
            subprocess.run(['rm', csv_path])

    # Getting rid of unnecessary video files
    print('Deleting {} clipped videos...'.format(args.video))
    subprocess.run(['rm', '-rf', pfm.local_files[video_name]])

if args.annotate:
    # Annotating the queried video file using the predicted boxes and labels
    print('Starting the video annotation process...')
    video_ann = VideoAnnotation(args.pid, video_path, args.video, csv_name, pfm)
    video_ann.annotate()

print('Process complete!')

if args.sync:
    sync_detection_dir()

print("Start Time (Full): ", s)
print("End Time (Full): ", ctime(time.time()))
