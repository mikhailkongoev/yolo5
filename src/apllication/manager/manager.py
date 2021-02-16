import time
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.apllication.config import Configuration
from src.apllication.cv_utils.drawing import draw_rectangle, put_text
from src.apllication.cv_utils.io import get_video_capture, get_video_writer, video_retrieve_frame, video_write_frame
from src.apllication.tracking.sort import Sort

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
        # TODO add nms threshold option
        tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)

        self.recognize(
            cap=cap,
            video_meta=meta,
            out=out,
            ball_detector=ball_detector,
            tracker=tracker,
            process_nth_frame=1,
        )

    def recognize(self,
                  cap,
                  video_meta,
                  out,
                  ball_detector,
                  tracker,
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

                # Filter boxes by class
                person_boxes, table_confs = self.filter_class(labels=labels, boxes=boxes, confs=confs,
                                                              label_to_filter='person')

                # Tracker
                tracks = tracker.update(person_boxes).astype(int)
                tracked_balls_boxes = tracks[:, :4]
                track_ids = tracks[:, 4]

                for box, track_id in zip(tracked_balls_boxes, track_ids):
                    draw_rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    put_text(frame, str(track_id), (box[0], box[1]))
                    if track_id not in history:
                        history[track_id] = [[box], count_frame]
                    else:
                        history[track_id][0].append(box)
                        history[track_id][1] = count_frame

                video_write_frame(out, frame)

            pbar.update()

        cap.release()
        out.release()
