import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join, exists
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pdb, matplotlib, os, argparse, cv2
from matplotlib.animation import FuncAnimation
from CichlidDetection.Classes.FileManager import FileManager

pd.set_option('display.expand_frame_repr', False)

class Animation:

    def __init__(self):
        self.fm = FileManager()
        self.img_dir = os.path.join(self.fm.local_files['detect_project'], "Frames")
        self.detect_dir = self.fm.local_files['detect_project']


    def map(self, box):
        if len(box) == 0:
            return [None]
        else:
            mapping = []
            for i in range(len(box)):
                try:
                    mapping.append(i)
                except ValueError:
                    mapping.append(None)
            return mapping


    def convert_pos(self, x1, y1, x2, y2):
        return [x1, y1, x2 - x1, y2 - y1]


    def animated_learning(self):
        """for a single frame, successively plot the predicted boxes and labels at each epoch to create an animation"""

        # collect a good set of frames to animate
        self.fm = FileManager()
        frame_dir =
        dir = '/Users/rhiyasharma/Documents/_McGrathLab/CD_work/frames'
        detect_dir = '/Users/rhiyasharma/Documents/_McGrathLab/CD_work'

        df = pd.read_csv('/Users/rhiyasharma/Documents/_McGrathLab/CD_work/csv/detections_new.csv')
        df = df[['Framefile', 'boxes', 'labels', 'scores']]
        df[['boxes', 'labels']] = df[['boxes', 'labels']].applymap(lambda x: eval(x))
        df['order'] = df.apply(lambda x: int(x.Framefile.split('.')[0].split('_')[1]), axis=1)
        df = df.sort_values(by=['order']).reset_index()
        df['len'] = df['boxes'].apply(lambda x: len(x))
        dd = df[df['len'] > 1]
        # df.to_csv(os.path.join(detect_dir, 'detections_new_Ordered.csv'))

        # build up the animation
        fig = plt.figure()
        max_detections = 5
        ax = fig.add_subplot(111)

        boxes = [Rectangle((0, 0), 0, 0, fill=False) for _ in range(max_detections)]

        def animate(i):
            new_frame = df.iloc[i].Framefile
            new_img = mpimg.imread(os.path.join(dir, new_frame))
            img = plt.imshow(new_img)
            ax = plt.gca()
            label_preds = df.labels[i]
            label_preds = (label_preds + ([0] * max_detections))[:5]
            box_preds = df.boxes[i]
            box_preds = [convert_pos(*p) for p in box_preds]
            box_preds = (box_preds + ([[0, 0, 0, 0]] * max_detections))[:5]
            color_lookup = {0: 'None', 1: '#FF1493', 2: '#00BFFF'}
            for j in range(5):
                boxes[j].set_xy([box_preds[j][0], box_preds[j][1]])
                boxes[j].set_width(box_preds[j][2])
                boxes[j].set_height(box_preds[j][3])
                boxes[j].set_edgecolor(color_lookup[label_preds[j]])

            for box in boxes:
                print(box)
                ax.add_patch(box)

            img.set_array(new_img)

            print('Completed ', new_frame)

            return [img]

        anim = FuncAnimation(fig, animate, frames=len(df), blit=True, interval=200, repeat=False)
        # # # ax.imshow(im, zorder=0)
        anim.save(os.path.join(detect_dir, 'detections_new1.mp4'), writer='imagemagick')
        plt.close('all')
