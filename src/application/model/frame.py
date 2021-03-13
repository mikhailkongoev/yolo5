from typing import List

import numpy as np

from model.bbox import Bbox


class Frame:
    def __init__(self, frame: np.ndarray, bboxes: List[Bbox]):
        self.frame = frame
        self.bboxes = bboxes
