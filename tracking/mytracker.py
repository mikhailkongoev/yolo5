import cv2


import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment

# from kalmanfilter import KalmanFilter
# from openvino_detectors.OpenvinoReidClassifier import OpenvinoReidentificationPlayer
from tracking.kalmanfilter import KalmanFilter
from tracking.linear_assignment_ import linear_assignment
import json
OUTSIDE_CONTOUR = -1
INSIDE_CONTOUR = 1
BORDER = 0
SOFT_IOU_ADD = 0.25

# def iou(bb_test, bb_gt, inc_size):
#     """
#   Computes IUO between two bboxes in the form [x1,y1,x2,y2]
#   """
#     if (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) > inc_size * (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]):
#         return 0
#     xx1 = np.maximum(bb_test[0], bb_gt[0])
#     yy1 = np.maximum(bb_test[1], bb_gt[1])
#     xx2 = np.minimum(bb_test[2], bb_gt[2])
#     yy2 = np.minimum(bb_test[3], bb_gt[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
#               + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
#     return o


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = x[2]
    h = x[3]
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((4, 1))


class Box:
    def __init__(self, track_id, left, bottom, right, top, frame, phantom):
        self.track_id = track_id
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.frame = frame
        self.phantom = phantom


class Track:
    def __init__(self, track_id, box, vx, vy, ax, ay):
        self.track_id = track_id
        self.boxes = [box]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, vx, 0, 0, 0],
             [0, 1, 0, 0, 0, vy, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, ax, 0],
             [0, 0, 0, 0, 0, 1, 0, ay],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(np.array([box.left, box.bottom, box.right, box.top]))
        self.kf.predict()
        self.last_frame = box.frame
        self.real_boxes = 1
        self.counted = False
        self.phantom_boxes = 0
        self.max_phantom_boxes = 0

    def update_phantom(self):
        predicted = self.kf.x
        b = convert_x_to_bbox(predicted)
        new_box = Box(self.track_id, b[0], b[1], b[2], b[3], self.last_frame, True)
        self.last_frame += 1
        self.boxes.append(new_box)
        self.kf.update(predicted[:4])
        self.kf.predict()
        self.phantom_boxes += 1
        if self.phantom_boxes > self.max_phantom_boxes:
            self.max_phantom_boxes = self.phantom_boxes

    def update_real(self, box, non_decrease):
        if len(self.boxes) > 0:
            prev = self.boxes[-1]
            ratio = (box.right - box.left) * (box.top - box.bottom) / ((prev.right - prev.left) * (prev.top - prev.bottom))
            if ratio < non_decrease:
                predicted = convert_x_to_bbox(self.kf.x[:4])
                box = Box(self.track_id, predicted[0], predicted[1], predicted[2], predicted[3], self.last_frame, True)

        self.boxes.append(box)
        self.real_boxes += 1
        self.phantom_boxes = 0
        self.last_frame = box.frame
        self.kf.update(convert_bbox_to_z(np.array([box.left, box.bottom, box.right, box.top])))
        self.kf.predict()

    def get_prediction(self):
        return convert_x_to_bbox(self.kf.x[:4])

    def get_max_phantoms(self):
        return self.max_phantom_boxes


class PhantomSortTracker:
    def __init__(self, min_x, min_y, max_x, max_y, detection_thresh, nms_thresh,
                 detections_count, track_creation_score, min_phantom_omit, max_phantom_omit,
                 phantom_coef, non_decrease, inc_size, vx, vy, ax, ay, static_coef, soft_iou_k):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.detection_thresh = detection_thresh
        self.nms_thresh = nms_thresh
        self.detections_count = detections_count
        self.track_creation_score = track_creation_score
        self.min_phantom_omit = min_phantom_omit
        self.max_phantom_omit = max_phantom_omit
        self.phantom_coef = phantom_coef
        self.non_decrease = non_decrease
        self.inc_size = inc_size
        self.static_coef = static_coef
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.tracks = []
        self.next_id = 0
        self.objects = 0
        self.result = 0
        self.soft_iou_k = soft_iou_k
        # self.reid = OpenvinoReidentificationPlayer("/home/sergej/Digital Drill/1_jacket/no_jacket")

    @classmethod
    def from_json(cls, config_path):
        config = json.load(open(config_path))
        return PhantomSortTracker(min_x=config["min_x"], min_y=config["min_y"], max_x=config["max_x"], max_y=config["max_y"],
                                  detection_thresh=config["detection-thresh"], nms_thresh=config["nms-thresh"],
                                  detections_count=config["detections-count"], track_creation_score=config["track-creation"],
                                  min_phantom_omit=config["min-phantom"], max_phantom_omit=config["max-phantom"],
                                  phantom_coef=config["phantom-coef"], non_decrease=config["non-decrease"],
                                  vx=config["vx"], vy=config["vy"], ax=config["ax"], ay=config["ay"],
                                  inc_size=config["inc-size"],
                                  static_coef=config["static-coef"],
                                  soft_iou_k = config["soft-iou-k"])

    def iou(self, a, b):
        w_a = a[2] - a[0]
        w_b = b[2] - b[0]
        h_a = a[3] - a[1]
        h_b = b[3] - b[1]
        ratio = (w_a * h_a) / (w_b * h_b)
        if 1 / self.inc_size <= ratio <= self.inc_size:
            x1 = np.maximum(a[0], b[0])
            y1 = np.maximum(a[1], b[1])
            x2 = np.minimum(a[2], b[2])
            y2 = np.minimum(a[3], b[3])
            w_c = np.maximum(0., x2 - x1)
            h_c = np.maximum(0., y2 - y1)
            s_c = w_c * h_c
            if s_c > 0:
                return SOFT_IOU_ADD + (1 - SOFT_IOU_ADD) * s_c / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - s_c)
            elif self.soft_iou_k > 0:
                dx = np.abs((a[0] + a[2]) - (b[0] + b[2])) / 2
                dy = np.abs((a[1] + a[3]) - (b[1] + b[3])) / 2
                soft_iou = SOFT_IOU_ADD * (1 - max(dx / (w_a + w_b), dy / (h_a + h_b)) / self.soft_iou_k)
                if soft_iou > 0:
                    return soft_iou
        return 0

    def update(self, detections, frame):
        if len(detections) > 0:
            iou_matrix = np.zeros((len(detections), len(self.tracks) + len(detections)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = -self.iou(det, track.get_prediction())
                    if iou_matrix[d, t] < 0:
                        n = 0
                        while n < len(track.boxes) - 1 and not track.boxes[-1 - n].phantom:
                            n += 1
                        if n > 0:
                            iou_matrix[d, t] -= 1.0
                if det[-1] < self.track_creation_score:
                    iou_matrix[d, len(self.tracks) + d] = +self.detection_thresh
                else:
                    iou_matrix[d, len(self.tracks) + d] = -self.detection_thresh
            matched_indices = linear_assignment(iou_matrix)
            for row in matched_indices:
                b = detections[row[0]]
                if row[1] >= len(self.tracks):
                    id = self.next_id
                    self.next_id += 1
                    self.tracks.append(Track(id, Box(id, b[0], b[1], b[2], b[3], frame, False),
                                             self.vx, self.vy, self.ax, self.ay))
                elif iou_matrix[row[0], row[1]] < 0:
                    track = self.tracks[row[1]]
                    box = Box(track.track_id, b[0], b[1], b[2], b[3], frame, False)
                    track.update_real(box, self.non_decrease)
                    if not track.counted and track.real_boxes >= self.detections_count:
                        self.objects += 1
                        track.counted = True

        active_tracks = []
        colored_boxes = []
        for track in self.tracks:
            if track.last_frame < frame:
                track.update_phantom()
                phantom_threshold = np.minimum(self.max_phantom_omit,
                                               np.maximum(self.min_phantom_omit,
                                                          self.phantom_coef * track.get_max_phantoms()))
                box = track.boxes[-1]
                if track.phantom_boxes > phantom_threshold \
                        or box.left > self.max_x \
                        or box.right < self.min_x \
                        or box.top < self.min_y \
                        or box.bottom > self.max_y:
                   print("remove track " + str(track.track_id))
                else:
                    active_tracks.append(track)
            else:
                active_tracks.append(track)
            box = track.boxes[-1]
            if box.phantom == 0:
                colored_box = [int(box.left), int(box.bottom), int(box.right), int(box.top), int(track.track_id),
                               int(box.phantom)]
                colored_boxes.append(colored_box)
        self.tracks = active_tracks

        return colored_boxes

