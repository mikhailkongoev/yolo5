class Bbox:
    def __init__(self, class_name: str, top: int, left: int, bottom: int, right: int):
        self.class_name = class_name
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right