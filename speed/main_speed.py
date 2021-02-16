import time

import numpy as np
import torch
from tqdm import tqdm

from cv_utils.drawing import draw_rectangle, put_text
from cv_utils.io import get_video_capture, get_video_writer, video_retrieve_frame, video_write_frame
from tracking.sort import Sort

LABEL2IDX = {'person': 0}

def filter_class(labels: np.ndarray, boxes: np.ndarray, confs: np.ndarray, label_to_filter: str):
    label_idx = LABEL2IDX[label_to_filter]
    cond = labels == label_idx

    label_boxes = boxes[cond]
    label_confs = confs[cond]

    return label_boxes, label_confs

def main():
    cap, meta = get_video_capture('/home/sergej/Documents/action/WhatsApp Video 2021-01-22 at 17.01.09.mp4')
    out = get_video_writer(
        "out_whatsapp.mp4", meta,
        fps=meta['fps'],
    )
    ball_detector = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path_or_model='/home/sergej/PycharmProjects/os2d/eff_detector/yolov5s.pt',
                                   )
    ball_detector = ball_detector.to(torch.device("cuda"))
    # TODO add nms threshold option
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)

    recognize(
        cap=cap,
        video_meta=meta,
        out=out,
        ball_detector=ball_detector,
        tracker=tracker,
        process_nth_frame=1,
    )


def recognize(
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
            person_boxes, table_confs = filter_class(labels=labels, boxes=boxes, confs=confs, label_to_filter='person')

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


if __name__ == '__main__':
    main()
