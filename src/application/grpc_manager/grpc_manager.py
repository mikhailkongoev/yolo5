import logging
from concurrent import futures

import grpc
import numpy as np

from grpc_stubs import yolo_pb2_grpc, yolo_pb2
from manager.manager import Manager

WAIT_SEC = 10


class GrpcInterface(yolo_pb2_grpc.YOLOServicer):
    def __init__(self, manager: Manager):
        self.manager = manager

    def ProcessFrame(self, request: yolo_pb2.Frame, context) -> yolo_pb2.InferenceResult:
        try:
            input_frame = np.frombuffer(request.input_frame, np.uint8).reshape((request.height, request.width, 3))
            frame = self.manager.process_frame(input_frame)
            height, width, channels = frame.frame.shape
            message_image = np.ndarray.tobytes(frame.frame)
            print('Successfully handled request for frame with num: ' + str(request.frame_num))
            return yolo_pb2.InferenceResult(result_frame=message_image, bbox=frame.bboxes, height=height, width=width)
        except RuntimeError:
            print('Error during handling request with frame num: ' + str(request.frame_num))


class GrpcManager:
    def __init__(self, manager: Manager):
        self.manager = manager
        # grpc
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.manager.config.grpc_workers))
        yolo_pb2_grpc.add_YOLOServicer_to_server(GrpcInterface(self.manager), self.server)
        self.server.add_insecure_port(f'{self.manager.config.grpc_ip}:{self.manager.config.grpc_port}')
        logging.info(f'gRPC server is UP at {self.manager.config.grpc_ip}:{self.manager.config.grpc_port}')
        self.server.start()
        self.server.wait_for_termination()

    def stop(self):
        try:
            self.server.stop(WAIT_SEC).wait()
        except RuntimeError:
            pass
