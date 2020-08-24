import sys
import os
import cv2
import pandas as pd
from os.path import join
from CichlidDetection.Classes.FileManager import FileManager
from CichlidDetection.Classes.FileManager import ProjectFileManager


def convert_pos(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return [(x1, y1), (x2, y2)]


class VideoAnnotation:

    def __init__(self, pid, video_path, video, csv_file, *args):

        self.fm = FileManager()
        for i in args:
            self.pfm = i
        self.detection_dir = self.fm.local_files['detection_dir']
        self.video = video_path
        self.video_name = video.split('.')[0]
        self.ann_video_name = 'annotated_' + video.split('.')[0]
        self.csv_file_path = os.path.join(self.detection_dir, csv_file)

    def annotate(self):
        """for a each frame, successively plot the predicted boxes and labels to create a video"""
        df = pd.read_csv(self.csv_file_path)
        df[['boxes', 'labels']] = df[['boxes', 'labels']].applymap(lambda x: eval(x))
        df['order'] = df.apply(lambda x: int(x.Framefile.split('.')[0].split('_')[1]), axis=1)

        cap = cv2.VideoCapture(self.video)
        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        result = cv2.VideoWriter(os.path.join(self.pfm.local_files['detection_dir'], self.ann_video_name), cv2.VideoWriter_fourcc(*"mp4v"), 5, size)

        count = 0
        for i in range(vid_len):
            ret, frame = cap.read()
            if not ret:
                print("VideoError: Couldn't read frame ", count)
                break
            else:
                label_preds = df.labels[i]
                box_preds = df.boxes[i]
                box_preds = [convert_pos(*p) for p in box_preds]
                if len(label_preds) > 0:
                    for j in range(len(label_preds)):
                        start, end = box_preds[0][0], box_preds[0][1]
                        color_lookup = {1: (255, 153, 255), 2: (255, 0, 0)}
                        cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color_lookup[label_preds[j]], 2)
                        result.write(frame)
                        print('Completed Annotated Frame {}'.format(count))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    result.write(frame)
                    print('Completed Frame {}'.format(count))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            count += 1

        cap.release()
        result.release()
        cv2.destroyAllWindows()

        print("The detection video was successfully saved")
