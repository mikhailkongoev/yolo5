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

        def parse_point(input_str):
            return tuple(int(k.strip()) for k in input_str[1:-1].split(','))

        if not os.path.exists('/.dockerenv'):
            env.read_env(path)

        self.input_video_path = read_variable('INPUT_VIDEO_PATH', default='input/input.mp4')
        self.output_video_path = read_variable('OUTPUT_VIDEO_PATH', default='output/output.mp4')
        self.model_path = read_variable('MODEL_PATH', default='models/yolo5s.pt')
        self.device = read_variable('DEVICE', default='cuda')
        self.markup_bot_left = parse_point(read_variable('MARKUP_BOT_LEFT', default='(75,75)'))
        self.markup_bot_right = parse_point(read_variable('MARKUP_BOT_RIGHT', default='(75,75)'))
        self.markup_top_right = parse_point(read_variable('MARKUP_TOP_RIGHT', default='(75,75)'))
        self.markup_top_left = parse_point(read_variable('MARKUP_TOP_LEFT', default='(75,75)'))
        self.markup_width = float(read_variable('MARKUP_WIDTH', default='1.0'))
        self.markup_height = float(read_variable('MARKUP_HEIGHT', default='1.0'))
        self.markup_visualize = bool(read_variable('MARKUP_VISUALIZE', default='True'))
