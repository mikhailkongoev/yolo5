import numpy as np
from ensemble_boxes import nms


def nms_boxes(boxes, confs, img_shape, labels):
    boxes[:, 0] /= img_shape[1]
    boxes[:, 1] /= img_shape[0]
    boxes[:, 3] /= img_shape[0]
    boxes[:, 2] /= img_shape[1]
    boxes, confs, labels = nms([boxes], [confs], [labels], iou_thr=0.01)
    boxes[:, 0] *= img_shape[1]
    boxes[:, 1] *= img_shape[0]
    boxes[:, 2] *= img_shape[1]
    boxes[:, 3] *= img_shape[0]
    boxes = boxes.astype(int)

    return boxes, confs, labels


def iou(box1, box2):
    """Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims([box1], 0)
    bb_test = np.expand_dims([box2], 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o[0, 0]
