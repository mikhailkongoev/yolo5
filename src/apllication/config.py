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
        self.device = read_variable('DEVICE', default='cuda')
