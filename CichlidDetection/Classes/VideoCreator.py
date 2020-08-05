import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from CichlidDetection.Classes.FileManager import FileManager

pd.set_option('display.expand_frame_repr', False)


def convert_pos(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]


class Animation:

    def __init__(self, pid, video_name, csv_file, *args):

        self.fm = FileManager()
        for i in args:
            self.pfm = i
        # collect a good set of frames to animate
        self.frame_dir = os.path.join(self.pfm.local_files['{}_dir'.format(pid)], "Frames")
        self.detection_dir = self.fm.local_files['detection_dir']
        self.video_name = video_name
        self.csv_file_path = os.path.join(self.detection_dir, csv_file)

    def animated_learning(self):
        """for a single frame, successively plot the predicted boxes and labels at each epoch to create an animation"""

        df = pd.read_csv(os.path.join(self.csv_file_path))
        df = df[['Framefile', 'boxes', 'labels', 'scores']]
        df[['boxes', 'labels']] = df[['boxes', 'labels']].applymap(lambda x: eval(x))
        df['order'] = df.apply(lambda x: int(x.Framefile.split('.')[0].split('_')[1]), axis=1)
        df = df.sort_values(by=['order']).reset_index()
        df['len'] = df['boxes'].apply(lambda x: len(x))

        # build up the animation
        fig = plt.figure()
        max_detections = 5
        ax = fig.add_subplot(111)

        boxes = [Rectangle((0, 0), 0, 0, fill=False) for _ in range(max_detections)]

        def animate(self, i):
            new_frame = df.iloc[i].Framefile
            new_img = mpimg.imread(os.path.join(self.frame_dir, new_frame))
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
        anim.save(os.path.join(self.detection_dir, '{}_detections.mp4'.format(self.video_name)), writer='imagemagick')
        plt.close('all')

        return '{}_detections.mp4'.format(self.video_name)
