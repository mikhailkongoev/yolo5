import time

import numpy as np
import torch

from model.bbox import Bbox
from model.frame import Frame
from src.application.config import Configuration
from ml_utils.bricks_utils import nms_boxes
from ml_utils.drawing_utils import draw_rectangle
from ml_utils.postprocessing_utils import filter_class, postprocessing


class Manager:

    def __init__(self, config: Configuration):
        self.config = config
        self.model = self.init_detector()

    def init_detector(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=self.config.model_path)
        model.to(torch.device(self.config.device))
        return model

    def process_frame(self, frame: np.ndarray) -> Frame:
        before = time.time()
        cur = self.model(frame)
        after = time.time()

        print("Frame processed for: " + str((after - before)) + ' sec.')
        yolo_preds = cur.xyxy[0].cpu().numpy()
        boxes = yolo_preds[:, :4]  # xmin, ymin, xmax, ymax
        confs = yolo_preds[:, 4]
        labels = yolo_preds[:, 5]

        if len(boxes):
            boxes, confs, labels = nms_boxes(boxes, confs, frame.shape, labels)
        brick_boxes, brick_confs = filter_class(labels, boxes, confs, 0)
        deffect_boxes, deffect_confs = filter_class(labels, boxes, confs, 1)
        deffect_boxes, deffect_confs = postprocessing(brick_boxes, deffect_boxes, deffect_confs)

        for box in brick_boxes:
            draw_rectangle(frame, box, (0, 255, 0), 2)
        for box in deffect_boxes:
            draw_rectangle(frame, box, (255, 0, 0), 3)

        return Frame(frame, [Bbox('Brick', bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in brick_boxes] +
                     [Bbox('Defect', bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in deffect_boxes])
