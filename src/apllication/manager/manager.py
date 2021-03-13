import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from src.apllication.config import Configuration
from src.apllication.cv_utils.drawing import draw_rectangle, put_text, draw_polygon
from src.apllication.cv_utils.io import get_video_capture, get_video_writer, video_retrieve_frame, video_write_frame


class Manager:

    def __init__(self, config: Configuration):
        self.initial_bboxes = []
        self.config = config
        self.labels = self.read_labels(self.config.labels_path)
        self.roi = self.read_polygon(self.config.roi_path)

        with open(self.config.labels_path, 'r') as file:
            for idx, line in enumerate(file):
                self.labels[idx] = line.strip()

    @staticmethod
    def read_labels(labels_path: str) -> Dict[int, str]:
        labels = dict()
        with open(labels_path, 'r') as file:
            for idx, line in enumerate(file):
                labels[idx] = line.strip()
        return labels

    @staticmethod
    def read_polygon(file_path: str) -> Polygon:
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                (x, y) = line.split(',')
                points.append((int(x), int(y)))
        return Polygon(points)

    def process(self):
        cap, meta = get_video_capture(self.config.input_video_path)
        out = get_video_writer(
            self.config.output_video_path, meta,
            fps=meta['fps'],
        )
        ball_detector = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path_or_model=self.config.model_path,
                                       )
        ball_detector = ball_detector.to(torch.device(self.config.device))
        print(ball_detector.module.names if hasattr(ball_detector, 'module') else ball_detector.names)

        self.recognize(
            cap=cap,
            video_meta=meta,
            out=out,
            ball_detector=ball_detector,
            process_nth_frame=1,
        )

    def recognize(self,
                  cap,
                  video_meta,
                  out,
                  ball_detector,
                  process_nth_frame=1,
                  ):
        pbar = tqdm(total=video_meta['n_frames'])
        history = {}
        start = 0
        for count_frame in range(video_meta['n_frames']):
            ret = cap.grab()
            if not ret:
                break

            if count_frame % process_nth_frame == 0:
                ret, frame = video_retrieve_frame(cap)

                # Balls detector
                before = time.time()
                cur = ball_detector(frame)
                start += time.time() - before
                yolo_preds = cur.xyxy[0].cpu().numpy()
                if count_frame % 20:
                    print("FPS")
                    print(count_frame / start)
                boxes = yolo_preds[:, :4]  # xmin, ymin, xmax, ymax
                confs = yolo_preds[:, 4]
                labels = yolo_preds[:, 5]

                # ROI
                draw_polygon(frame,
                             list(zip(self.roi.exterior.coords.xy[0], self.roi.exterior.coords.xy[1])),
                             color=(0, 0, 255), thickness=2)

                # for box, track_id, label in zip(tracked_balls_boxes, track_ids, config):
                for box, label in zip(boxes, labels):
                    if count_frame <= 3:
                        self.initial_bboxes.append(box)
                    draw_rectangle(frame, box, color=self.find_color(box, count_frame), thickness=2)
                    put_text(frame, self.labels[label] if label in self.labels else str(label),
                             (box[0], box[1]))
                    # if track_id not in history:
                    #     history[track_id] = [[box], count_frame]
                    # else:
                    #     history[track_id][0].append(box)
                    #     history[track_id][1] = count_frame

                video_write_frame(out, frame)

            pbar.update()

        cap.release()
        out.release()

    def find_color(self, box: List[int], count_frame: int) -> Tuple[int, int, int]:
        if count_frame <= 3:
            return 0, 255, 0
        if self.is_bbox_inside_roi(box) and self.find_appropriate_bbox(box, self.initial_bboxes) is None:
            return 255, 0, 0
        return 0, 255, 0

    def is_bbox_inside_roi(self, bbox: List[int]):
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        point = Point(x, y)
        return self.roi.contains(point)

    def find_appropriate_bbox(self, bbox: List[int], bboxes_list: List[List[int]]) -> Optional[List[int]]:
        for bbox_iter in bboxes_list:
            iou = self.calc_iou(bbox_iter, bbox)
            if iou > self.config.roi_initial_iou_threshold:
                return bbox_iter

        return None

    @staticmethod
    def calc_iou(gt_bbox, pred_bbox):
        """
        This function takes the predicted bounding box and ground truth bounding box and
        return the IoU ratio
        """
        x_tl_gt, y_tl_gt, x_br_gt, y_br_gt = gt_bbox
        x_tl_p, y_tl_p, x_br_p, y_br_p = pred_bbox

        if (x_tl_gt > x_br_gt) or (y_tl_gt > y_br_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_tl_p > x_br_p) or (y_tl_p > y_br_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_tl_p, x_br_p, y_tl_p,
                                 y_br_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if x_br_gt < x_tl_p or y_br_gt < y_tl_p or x_tl_gt > x_br_p or y_tl_gt > y_br_p:
            return 0.0

        gt_bbox_area = (x_br_gt - x_tl_gt + 1) * (y_br_gt - y_tl_gt + 1)
        pred_bbox_area = (x_br_p - x_tl_p + 1) * (y_br_p - y_tl_p + 1)

        x_top_left = np.max([x_tl_gt, x_tl_p])
        y_top_left = np.max([y_tl_gt, y_tl_p])
        x_bottom_right = np.min([x_br_gt, x_br_p])
        y_bottom_right = np.min([y_br_gt, y_br_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (gt_bbox_area + pred_bbox_area - intersection_area)

        return intersection_area / union_area
