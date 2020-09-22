import pandas as pd
from itertools import combinations
from CichlidDetection.Classes.FileManager import FileManager
from os.path import join

pd.set_option('display.expand_frame_repr', False)


def convert_pos(x1, y1, x2, y2):
    """ Convert coordinates to [(x1, y1),(x2, y2)] format

    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return [(x1, y1), (x2, y2)]


def mapper(elements):
    """ Create order of elements

    """
    if len(elements) == 0:
        return []
    else:
        mapping = []
        for i in range(len(elements)):
            try:
                mapping.append(i)
            except ValueError:
                mapping.append([])

    return mapping


def calc_iou(combination):
    """ Calculate intersection / union between two different sets of coordinates

    """
    a = combination[0]
    b = combination[1]
    if (len(a) == 0) and (len(b) == 0):
        return 1.0
    # find area of the box formed by the intersection of a and b
    xa, ya, xb, yb = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    # if the boxes do not intersect, short-circuit and return 0.0
    if intersection == 0:
        return 0.0
    # else, calculate the area of the union of box_a and box_b, and return intersection/union
    else:
        a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        union = float(a_area + b_area - intersection)
        iou = intersection / union
        return iou


def combos(box):
    """ Calculate different combinations made between multiple boxes within a frame

    """
    if len(box) == 0 or len(box) == 1:
        return []
    else:
        combo = list((i, j) for ((i, _), (j, _)) in combinations(enumerate(box), 2))

    return combo


def iou(box, frame):
    """ Calculate IOU between box_a and box_b

    """
    iou = []
    sets = list(combinations(box, 2))
    for i in range(len(sets)):
        val = calc_iou(sets[i])
        iou.append(val)

    return iou


def update_lists(comb, iou_score, map_index, scores, frame):
    new_map = map_index
    same = []
    update = []
    similar = []
    for i in range(len(comb)):
        if iou_score[i] != [] and iou_score[i] > 0.5:
            same.append(comb[i])

    for i in range(len(same)):
        similar.append((scores[same[i][0]], scores[same[i][1]]))

    for i in range(len(new_map)):
        for j in range(len(same)):
            if new_map[i] in same[j]:
                max_val = max(similar[j])
                max_el = scores.index(max_val)
                new_map[i] = same[j][same[j].index(max_el)]

    # if frame == 'Frame_14725.jpg':
    #     print(frame, same, scores)
    #     print('UPDATED: ', new_map)
    #     print('len: ', len(set(new_map)))

    return new_map


def num_fish(comb, iou_score, map_index, scores, frame, count):
    """ Calculate the actual number of fish within a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores, frame)
    count += 1
    print(count, len(set(edit_elements)))
    return count, len(set(edit_elements))


def update_boxes(comb, iou_score, map_index, scores, boxes, frame):
    """ Update the actual list of box coordinates in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores, frame)
    edit_elements = set(edit_elements)
    update_box = []
    for i in edit_elements:
        update_box.append(boxes[i])

    return update_box


def update_labels(comb, iou_score, map_index, scores, labels, frame):
    """ Update the actual list of labels in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores, frame)
    edit_elements = set(edit_elements)
    update_label = []
    for i in edit_elements:
        update_label.append(labels[i])

    return update_label


def update_scores(comb, iou_score, map_index, scores, frame):
    """ Update the actual list of scores of detections present in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores, frame)
    edit_elements = set(edit_elements)
    update_score = []
    for i in edit_elements:
        update_score.append(scores[i])

    return update_score


class Tracking:

    def __init__(self, *args):
        # for i in args:
        #     self.pfm = i
        self.fm = FileManager()

    def compare_frame_iou(self, csv_path):
        """ Create new columns which compares the iou scores between the boxes within a frame. Discarding all unnecessary columns.

        Args:
                csv_path (str): path to the detections csv file

        """
        df = pd.read_csv(csv_path)
        count= 0
        df[['boxes', 'labels', 'scores']] = df[['boxes', 'labels', 'scores']].applymap(lambda x: eval(x))
        df['map_index'] = df.apply(lambda x: mapper(x.boxes), axis=1)
        df['sets'] = df.apply(lambda x: combos(x.labels), axis=1)
        df['iou'] = df.apply(lambda x: iou(x.boxes, x.Framefile), axis=1)
        count, df['n_fish'] = zip(*df.apply(lambda x: num_fish(x.sets, x.iou, x.map_index, x.scores, x.Framefile, count), axis=1))
        # count, df['n_fish'] = df.applymap(lambda x: num_fish(x.sets, x.iou, x.map_index, x.scores, x.Framefile, count), axis=1)
        df['boxes_final'] = df.apply(lambda x: update_boxes(x.sets, x.iou, x.map_index, x.scores, x.boxes, x.Framefile),
                                     axis=1)
        df['labels_final'] = df.apply(
            lambda x: update_boxes(x.sets, x.iou, x.map_index, x.scores, x.labels, x.Framefile), axis=1)
        df['scores_final'] = df.apply(lambda x: update_scores(x.sets, x.iou, x.map_index, x.scores, x.Framefile),
                                      axis=1)
        df.drop(['boxes', 'labels', 'map_index', 'scores', 'sets', 'iou'], axis=1, inplace=True)

        # TODO: try to implement something like this - df['average_iou'], df['act_to_pred_map'] = zip(*df.apply(
        #             lambda x: self._calc_frame_iou(x.boxes_actual, x.boxes_predicted, map_boxes=True), axis=1))

        df.to_csv(join(self.fm.local_files['detection_dir'], 'updated_detections.csv'))

        return df

    def compare_row_iou(self, dd):
        """ Create new columns which compares the iou scores between the boxes within a frame. Discarding all unnecessary columns.

        Args:
                df (str): name of Dataframe containing the updated list

        """
        df = dd
        print(df)


fm = FileManager()
track = Tracking()
df = track.compare_frame_iou(join(fm.local_files['detection_dir'], 'MC6_5_10_0001_vid_detections.csv'))
# print(df)
