import os
import cv2
from os.path import join
from CichlidDetection.Classes.TrackingFish import Tracking
from CichlidDetection.Classes.FileManager import FileManager
# from CichlidDetection.Classes.FileManager import ProjectFileManager


def convert_pos(x1, y1, x2, y2):
    # convert coordinates to desired format for cv2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return [(x1, y1), (x2, y2)]


class VideoAnnotation:
    """ For each video successively plot the predicted boxes and labels to create a new annotated video

    Args:
            pid: Project ID
            video_path: Path specifying the location of the ~10h video
            video: Name of the video
            csv_file: Name of the csv file containing all the predicted boxes and labels
            *args: Project File Manager function

    """

    def __init__(self, pid, video_path, video, csv_file, *args):

        self.fm = FileManager()
        self.track = Tracking()
        for i in args:
            self.pfm = i
        self.detection_dir = self.fm.local_files['detection_dir']
        self.video = video_path
        self.video_name = video.split('.')[0]
        self.ann_video_name = 'annotated_' + pid + '_' + self.video_name + '_p2.mp4'
        self.csv_file_path = join(self.detection_dir, csv_file)

    def annotate(self):

        dd = self.track.diff_fish(self.csv_file_path)
        df = self.track.track_fish_row(dd)

        cap = cv2.VideoCapture(self.video)
        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        # font details - add frame name to the video frames
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        x, y = 1100, 150

        result = cv2.VideoWriter(os.path.join(self.detection_dir, self.ann_video_name), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10, size)

        # count = 0
        for i in range(vid_len):
            ret, frame = cap.read()
            if not ret:
                print("VideoError: Couldn't read frame ", i)
                break
            else:
                box_preds = df.boxes[i]
                box_preds = [convert_pos(*p) for p in box_preds]
                label_preds = df.labels[i]
                scores = df.scores[i]
                n_fish = df.n_fish[i]
                fish_ID = df.fish_ID[i]
                color_lookup = {1: (255, 153, 255), 2: (255, 0, 0)}
                font_text = 'Frame_{}.jpg'.format(i)
                cv2.putText(frame, font_text, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
                for j in range(len(fish_ID)):
                    start, end = box_preds[j][0], box_preds[j][1]
                    cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color_lookup[label_preds[j]], 2)
                    cv2.putText(frame, 'fish {}'.format(fish_ID[j]), (end[0] + 2, end[1] - 5), font, font_size, (0, 0, 0), 1,
                                cv2.LINE_AA)
                    result.write(frame)
                    print('Completed Annotating Frame {}'.format(i))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    font_text = 'Frame_{}.jpg'.format(i)
                    cv2.putText(frame, font_text, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
                    result.write(frame)
                    print('Completed Frame {}'.format(i))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


        cap.release()
        result.release()
        cv2.destroyAllWindows()

        print("The detection video was successfully saved")
        print("Location of the video: ", os.path.join(self.detection_dir, self.ann_video_name))
