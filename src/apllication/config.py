import os
from pathlib import Path
from typing import Optional, Union

from environs import Env


class Configuration:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        env = Env()
        if isinstance(path, Path):
            path = str(path.resolve())

        def read_variable(key, default):
            return os.getenv(key, default=env.str(key, default=default))

        if not os.path.exists('/.dockerenv'):
            env.read_env(path)

        self.input_video_path = read_variable('INPUT_VIDEO_PATH', default='input/input.mp4')
        self.output_video_path = read_variable('OUTPUT_VIDEO_PATH', default='output/output.mp4')
        self.model_path = read_variable('MODEL_PATH', default='models/yolo5s.pt')
        self.labels_path = read_variable('LABELS_PATH', default='config/config.txt')
        self.device = read_variable('DEVICE', default='cuda')

        self.roi_path = read_variable('ROI_PATH', default='config/roi_points.txt')
        self.roi_initial_iou_threshold = float(read_variable('ROI_INITIAL_IOU_THRESHOLD', default='0.85'))

