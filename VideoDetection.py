import os
import argparse
import subprocess
from CichlidDetection.Classes.Detector import Detector
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.FileManager import ProjectFileManager

# parse command line arguments
parser = argparse.ArgumentParser(description='To Detect Cichlids in Videos')
parser.add_argument('pid', type=str, metavar=' ', help='Project ID. Ex: MC6_5')
parser.add_argument('video', type=int, metavar = ' ', help='Run detection on specified video. Ex: 0005_vid.mp4')
parser.add_argument('-i','--download_images', type='store_true', metavar=' ', help='Download the full image directory')
parser.add_argument('-v','--download_video', action='store_true', metavar=' ', help='Download video')
args = parser.parse_args()

"""
Download videos from different projects and run them through the model to detect cichlids

Args:
    pid (str): project id
    download_images (bool): if True, download the full image directory for the specified project
    download_videos (bool): if True, download the all the mp4 files in Videos directory for the specified project
    video (str): specifies which video to download
"""

fm = FileManager()
pfm = ProjectFileManager(args.pid, fm, args.download_images, args.download_video, args.video)
de = Detector()
# video_path = fm.local_files['project_dir']
de.frame_detect(args.pid, )
