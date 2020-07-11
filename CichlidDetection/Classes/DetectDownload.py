import argparse, os, subprocess, pdb, matplotlib, sys
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import os, tarfile
from os.path import join
from PIL import Image
from matplotlib import rcParams
import numpy as np
import matplotlib.image as mpimg
import cv2
from CichlidDetection.Utilities.utils import run, make_dir
from CichlidDetection.Classes.FileManager import FileManager

class animation:

    def __init__(self):
        self.fm = FileManager()
        self.pid = 'MC6_5'

    def _make_dir(self, name, path):
        """update the self.local_files dict with {name: path}, and create the directory if it does not exist

        Args:
            name (str): brief file descriptor, to be used as key in the local_files dict
            path (str): local path of the directory to be created

        Returns:
            str: the path argument, unaltered
        """
        self.fm.local_files.update({name: make_dir(path)})
        return path

    def _locate_cloud_files(self):
        """locate the required files in Dropbox.

        Returns:
            string: cloud_master_dir, the outermost Dropbox directory that will be used henceforth
            dict: cloud_files, a dict of paths to remote files, keyed by brief descriptors
        """
        # establish the correct remote
        possible_remotes = run(['rclone', 'listremotes']).split()
        if len(possible_remotes) == 1:
            remote = possible_remotes[0]
        elif 'cichlidVideo:' in possible_remotes:
            remote = 'cichlidVideo:'
        elif 'd:' in possible_remotes:
            remote = 'd:'
        else:
            raise Exception('unable to establish rclone remote')

        # establish the correct path to the CichlidPiData directory
        root_dir = [r for r in run(['rclone', 'lsf', remote]).split() if 'McGrath' in r][0]
        cloud_master_dir = join(remote + root_dir, 'Apps', 'CichlidPiData')

        # locate essential, project non-specific files
        remoteDir = cloud_master_dir + '/{}/Frames/'.format(self.pid)
        remote_files = subprocess.run(['rclone', 'lsf', remoteDir], capture_output=True, encoding='utf-8')

        return cloud_master_dir, remoteDir, remote_files

    def download(self, remoteDir, file_list):
        file_list = files.stdout.split()
        imgs=[]
        npys=[]
        for file in file_list:
            if file.endswith('.npy'):
                npys.append(file)
        for file in file_list:
            if file.endswith('.jpg'):
                imgs.append(file)

        dir = self.fm.local_files['detection_dir']

        # Creating a project subdir
        subdir_pid = self.pid
        if not os.path.exists(dir + '/'+ subdir_pid):
            self._make_dir('detect_project', join(dir, subdir_pid))
        else:
            self.fm.local_files['detect_project'] = os.path.join(dir, subdir_pid)

        projDir = self.fm.local_files['detect_project']

        if not os.path.exists(projDir+'/'+'images'):
            os.mkdir(os.path.join(projDir, 'images'))

        if not os.path.exists(projDir+'/'+'npys'):
            os.mkdir(os.path.join(projDir, 'npys'))

        local_imgs = os.path.join(projDir, 'images')
        local_npys = os.path.join(projDir, 'npys')

        if len(os.listdir(local_imgs)) < 1800:
            print('downloading images')
            for img in imgs[1:1801]:
                if img not in local_imgs:
                    img_path = os.path.join(remoteDir, img)
                    subprocess.run(['rclone', 'copy', img_path, local_imgs], stderr=subprocess.PIPE)
                    print('downloaded {}'.format(img))

        print('downloaded all image files')

        if len(os.listdir(local_npys)) < 1800:
            print('downloading npy files')
            for npy in npys[1:1801]:
                if npy not in local_npys:
                    npy_path = os.path.join(remoteDir, npy)
                    subprocess.run(['rclone', 'copy', npy_path, local_npys], stderr=subprocess.PIPE)
                    print('downloaded {}'.format(npy))

        print('downloaded all npy files')



anim = animation()
master, img_dir, files = anim._locate_cloud_files()
anim.download(img_dir, files)
