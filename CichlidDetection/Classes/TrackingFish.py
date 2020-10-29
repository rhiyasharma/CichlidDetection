import pandas as pd
import numpy as np
import itertools
from CichlidDetection.Classes.FileManager import FileManager
from os.path import join

pd.set_option('display.expand_frame_repr', False)


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
        iou = 1.0
    # find area of the box formed by the intersection of a and b
    xa, ya, xb, yb = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    # if the boxes do not intersect, short-circuit and return 0.0
    if intersection == 0:
        iou = 0.0
    # else, calculate the area of the union of box_a and box_b, and return intersection/union
    else:
        a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        union = float(a_area + b_area - intersection)
        iou = intersection / union

    return iou


def combos(box):
    if len(box) == 0 or len(box) == 1:
        return []
    else:
        mapped = mapper(box)
        combo = list(itertools.combinations(mapped, 2))

    return combo


def iou(box, frame):
    """ Calculate IOU between box_a and box_b

    """
    iou = []
    # box = [convert_pos(*p) for p in box]
    sets = list(itertools.combinations(box, 2))
    for i in range(len(sets)):
        val = calc_iou(sets[i])
        iou.append(val)

    return iou


def update_lists(comb, iou_score, map_index, scores):
    new_map = map_index
    same = []
    remove_el = []
    score_comparison = []

    for i in range(len(map_index)):
        if scores[i] < 0.4:
            remove_el.append(i)

    for i in remove_el:
        new_map.remove(i)

    for i in range(len(comb)):
        if iou_score[i] != [] and iou_score[i] > 0.4:
            same.append(comb[i])

    for i in range(len(same)):
        score_comparison.append((scores[same[i][0]], scores[same[i][1]]))

    for i in range(len(new_map)):
        for j in range(len(same)):
            if new_map[i] in same[j]:
                max_val = max(score_comparison[j])
                max_el = scores.index(max_val)
                new_map[i] = same[j][same[j].index(max_el)]

    return new_map


def num_fish(comb, iou_score, map_index, scores):
    """ Calculate the actual number of fish within a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores)
    return len(set(edit_elements))


def update_boxes(comb, iou_score, map_index, scores, boxes):
    """ Update the actual list of box coordinates in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores)
    edit_elements = set(edit_elements)
    update_box = []
    for i in edit_elements:
        update_box.append(boxes[i])

    return update_box


def update_labels(comb, iou_score, map_index, scores, labels):
    """ Update the actual list of labels in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores)
    edit_elements = set(edit_elements)
    update_label = []
    for i in edit_elements:
        update_label.append(labels[i])

    return update_label


def update_scores(comb, iou_score, map_index, scores):
    """ Update the actual list of scores of detections present in a frame

    """
    edit_elements = update_lists(comb, iou_score, map_index, scores)
    edit_elements = set(edit_elements)
    update_score = []
    for i in edit_elements:
        update_score.append(scores[i])

    return update_score


def iou_list(box1, box2, calc_iou_score=False):
    """ Calculate the iou between boxes between two different frames
    """

    if len(box1) == 0 or len(box2) == 0:
        combo = []
    else:
        combo = list(itertools.product(box1, box2))

    if calc_iou_score:
        ious = []
        for i in range(len(combo)):
            val = calc_iou(combo[i])
            ious.append(val)
        return ious
    else:
        return combo


def index_combos(box1, box2):
    map1 = mapper(box1)
    map2 = mapper(box2)

    index_combo = iou_list(map1, map2, calc_iou_score=False)

    return index_combo


def track_id(combo, n_fish, iou_score):
    new_map = combo
    same_fish = []
    sorted_map=[]
    f_id = []
    if len(iou_score) == 0 and n_fish == 0:
        return f_id
    elif len(iou_score) == 0 and n_fish != 0:
        for i in range(n_fish):
            f_id.append(i + 1)
        return f_id
    elif len(iou_score) > 0:
        sorted_map = [x for _,x in sorted(zip(iou_score, new_map), reverse=True)]
        for i in range(len(iou_score)):
            if iou_score[i] > 0.5 and len(same_fish) < n_fish:
                same_fish.append(new_map[i])

    for i in range(len(same_fish)):
        if same_fish[i][0] == same_fish[i][1]:
            f_id.append(i + 1)
        else:
            f_id.insert(0, i + 1)

    if len(f_id) > len(n_fish):


    return f_id


class Tracking:

    def __init__(self, *args):
        # for i in args:
        #     self.pfm = i
        self.fm = FileManager()

    def diff_fish(self, csv_path):
        """ Create new columns which compares the iou scores between the boxes within a frame. Discarding all unnecessary columns.

        Args:
                csv_path (str): path to the detections csv file

        """
        df = pd.read_csv(csv_path)
        count = 0
        df[['boxes', 'labels', 'scores']] = df[['boxes', 'labels', 'scores']].applymap(lambda x: eval(x))
        df['map_index'] = df.apply(lambda x: mapper(x.boxes), axis=1)
        df['sets'] = df.apply(lambda x: combos(x.boxes), axis=1)
        df['iou'] = df.apply(lambda x: iou(x.boxes, x.Framefile), axis=1)
        df['n_fish'] = df.apply(lambda x: num_fish(x.sets, x.iou, x.map_index, x.scores), axis=1)
        df['boxes'] = df.apply(
            lambda x: update_boxes(x.sets, x.iou, x.map_index, x.scores, x.boxes),
            axis=1)
        df['labels'] = df.apply(
            lambda x: update_labels(x.sets, x.iou, x.map_index, x.scores, x.labels), axis=1)
        df['scores'] = df.apply(
            lambda x: update_scores(x.sets, x.iou, x.map_index, x.scores),
            axis=1)
        df.drop(['map_index', 'sets', 'iou'], axis=1, inplace=True)
        # df.to_csv(join(self.fm.local_files['detection_dir'], 'updated_detections.csv'))

        return df

    def track_fish_row(self, dd):
        """ Create new columns which compares the iou scores between the corresponding boxes in two different frames.

        Args:
                dd: Dataframe containing the updated labels

        """
        df = dd
        # df = pd.read_csv(join(self.fm.local_files['detection_dir'], 'updated_detections.csv'))
        # df[['boxes', 'labels', 'scores']] = df[['boxes', 'labels', 'scores']].applymap(lambda x: eval(x))
        df['box_tracking'] = df['boxes'].shift(periods=1)
        # df['nfish_track'] = df['n_fish'].shift(periods=1)
        df['box_tracking'] = [list() if x is np.NaN else x for x in df['box_tracking']]
        # df.nfish_track.replace(np.NaN, 1.0, inplace=True)
        # df['nfish_track'] = df['nfish_track'].astype(int)
        df['sets'] = df.apply(lambda x: index_combos(x.boxes, x.box_tracking), axis=1)
        df['iou'] = df.apply(lambda x: iou_list(x.boxes, x.box_tracking, calc_iou_score=True), axis=1)
        df['fish_ID'] = df.apply(lambda x: track_id(x.sets, x.n_fish, x.iou), axis=1)
        df.to_csv(join(self.fm.local_files['detection_dir'], 'updated_detections.csv'))
        df.drop(['box_tracking', 'sets', 'iou'], axis=1, inplace=True)
        # print(df)
        return df


# fm = FileManager()
# track = Tracking()
# # dd = track.diff_fish('/Users/rhiyasharma/Downloads/Book1.csv')
# dd = track.diff_fish(join(fm.local_files['detection_dir'], 'MC6_5_10_0001_vid_detections.csv'))
#
# track.track_fish_row(dd)

