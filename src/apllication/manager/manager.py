import time
from collections import OrderedDict
from math import sqrt
from typing import Tuple, Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.apllication.config import Configuration
from src.apllication.cv_utils.drawing import draw_rectangle, put_text
from src.apllication.cv_utils.io import get_video_capture, get_video_writer, video_retrieve_frame, video_write_frame
from src.apllication.tracking.sort import Sort, AbstractTracker

LABEL2IDX = {'person': 0}


class Manager:

    def __init__(self, config: Configuration):
        self.config = config

    @staticmethod
    def filter_class(labels: np.ndarray, boxes: np.ndarray, confs: np.ndarray, label_to_filter: str) \
            -> Tuple[np.ndarray, np.ndarray]:
        label_idx = LABEL2IDX[label_to_filter]
        cond = labels == label_idx

        label_boxes = boxes[cond]
        label_confs = confs[cond]

        return label_boxes, label_confs

    def run(self):
        cap, meta = get_video_capture(self.config.input_video_path)
        writer = get_video_writer(self.config.output_video_path, meta, fps=meta['fps'])
        detector = self.load_detector()
        # TODO add nms threshold option
        tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)

        self.process_video(
            cap=cap,
            video_meta=meta,
            out=writer,
            ball_detector=detector,
            tracker=tracker,
            process_nth_frame=1,
        )

    def load_detector(self):
        ball_detector = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path_or_model=self.config.model_path,
                                       )
        ball_detector = ball_detector.to(torch.device(self.config.device))

        return ball_detector

    def visualize_markup(self, frame: np.ndarray):
        blue_color = (0, 0, 255)
        frame = cv2.circle(frame, self.config.markup_bot_left, radius=5, color=blue_color, thickness=-1)
        frame = cv2.circle(frame, self.config.markup_bot_right, radius=5, color=blue_color, thickness=-1)
        frame = cv2.circle(frame, self.config.markup_top_right, radius=5, color=blue_color, thickness=-1)
        frame = cv2.circle(frame, self.config.markup_top_left, radius=5, color=blue_color, thickness=-1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(self.config.markup_width), tuple(
            map(lambda x: int((x[0] + x[1]) / 2),
                zip(self.config.markup_bot_left, self.config.markup_bot_right))),
                    font, 0.5, blue_color, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.config.markup_height), tuple(
            map(lambda x: int((x[0] + x[1]) / 2),
                zip(self.config.markup_bot_left, self.config.markup_top_left))),
                    font, 0.5, blue_color, 2, cv2.LINE_AA)

    @staticmethod
    def get_camera_perspective(frame: np.ndarray, src_points: List[Tuple[int, int]]):
        image_h = frame.shape[0]
        image_w = frame.shape[1]
        src = np.float32(np.array(src_points))
        dst = np.float32([[0, image_h], [image_w, image_h], [0, 0], [image_w, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        return M, M_inv

    def process_video(self,
                      cap: cv2.VideoCapture,
                      video_meta: Dict,
                      out: cv2.VideoWriter,
                      ball_detector,
                      tracker: AbstractTracker,
                      process_nth_frame: int = 1,
                      ):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pbar = tqdm(total=video_meta['n_frames'])
        history = {}
        start = 0
        reading_first_frame = True
        for count_frame in range(video_meta['n_frames']):
            ret = cap.grab()
            if not ret:
                break

            if count_frame % process_nth_frame == 0:
                ret, frame = video_retrieve_frame(cap)

                if reading_first_frame:
                    self.M, self.Minv = self.get_camera_perspective(frame, [self.config.markup_bot_left,
                                                                            self.config.markup_bot_right,
                                                                            self.config.markup_top_left,
                                                                            self.config.markup_top_right])
                    reading_first_frame = False

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

                # Filter boxes by class
                person_boxes, table_confs = self.filter_class(labels=labels, boxes=boxes, confs=confs,
                                                              label_to_filter='person')

                # Tracker
                tracks = tracker.update(person_boxes).astype(int)
                tracked_balls_boxes = tracks[:, :4]
                track_ids = tracks[:, 4]

                person_centers = list(
                    map(lambda bbox: [int((bbox[0] + bbox[2]) / 2), int(bbox[3])], tracked_balls_boxes))

                person_centers = list(
                    map(lambda pts: cv2.perspectiveTransform(np.array([[pts]], dtype='float32'), self.M)[0][0],
                        person_centers))

                person_centers = list(map(lambda point: ((point[0] / width) * self.config.markup_width,
                                                         (point[1] / height) * self.config.markup_height),
                                          person_centers))

                for box, track_id, center_2d in zip(tracked_balls_boxes, track_ids, person_centers):
                    draw_rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    # put_text(frame, str(track_id), (box[0], box[1]))
                    # put_text(frame, str(center_2d), (box[0], box[1]))

                    if track_id not in history:
                        history[track_id] = OrderedDict()
                    history[track_id][count_frame] = {'bbox': box, 'coord_2d': center_2d}
                    speed_10 = self.evaluate_speed(history=history[track_id],
                                                   last_frame_num=count_frame,
                                                   for_frames_amount=10)
                    speed_11 = self.evaluate_speed(history=history[track_id],
                                                   last_frame_num=count_frame,
                                                   for_frames_amount=11)
                    speed_12 = self.evaluate_speed(history=history[track_id],
                                                   last_frame_num=count_frame,
                                                   for_frames_amount=12)
                    speed_13 = self.evaluate_speed(history=history[track_id],
                                                   last_frame_num=count_frame,
                                                   for_frames_amount=13)
                    speed_14 = self.evaluate_speed(history=history[track_id],
                                                   last_frame_num=count_frame,
                                                   for_frames_amount=14)

                    speed = (speed_10 + speed_11 + speed_12 + speed_13 + speed_14) / 5
                    history[track_id][count_frame]['speed'] = speed
                    put_text(frame, str("%.3f".format() % speed) + 'm/s', (box[0], box[1]))

                if self.config.markup_visualize:
                    self.visualize_markup(frame)

                video_write_frame(out, frame)

                pbar.update()

        cap.release()
        out.release()

    def evaluate_speed(self, history: Dict[int, Dict], last_frame_num: int, for_frames_amount: int) -> float:
        recent_history = {k: history[k] for k in range(last_frame_num - for_frames_amount, last_frame_num + 1) if
                          k in history}
        frame_before = recent_history[min(recent_history.keys())]
        frame_after = recent_history[max(recent_history.keys())]

        coord_before = frame_before['coord_2d']
        coord_after = frame_after['coord_2d']

        return sqrt(
            (coord_after[0] - coord_before[0]) ** 2 + (coord_after[1] - coord_before[1]) ** 2) * 25 / for_frames_amount
