import cv2
import grpc
import numpy as np

import yolo_pb2_grpc
from src.application.grpc_stubs.yolo_pb2 import Frame

channel = grpc.insecure_channel('localhost:50051')
stub = yolo_pb2_grpc.YOLOStub(channel)

input_frame = cv2.imread('input/1.jpg')
height, width, channels = input_frame.shape
message_image = np.ndarray.tobytes(input_frame)
request = Frame(input_frame=message_image, frame_num=42, height=height, width=width)
response = stub.ProcessFrame(request)
print(response)
print(response.bbox)
