"""
The main file for serving the encoded model using the model encoder.
"""

# module imports
import grpc
from concurrent import futures
import time

# import Keras model generator
from tensorflow_manager.model.model import ModelGenerator

# import protobuf created files
import tensorflow_manager.proto.python.service.service_pb2 as pb2
import tensorflow_manager.proto.python.service.service_pb2_grpc as pb2_grpc

# importing encoder to encode model, and send
from encoder import ProtoEncoder


class ModelEncodeService(pb2_grpc.ModelEncodeServicer):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.encoder = ProtoEncoder(self.model, "adam", None, "accuracy")

    def GetEncodedModel(self, request, context):
        start_time = time.time()
        encoded_model = self.encoder.encode_model()
        result = {'model': encoded_model, 'received': True}
        end_time = time.time()
        print(f"Time taken for returning response - {end_time - start_time}")
        return pb2.ModelResponse(**result)


def serve():
    net = ModelGenerator().get_model()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ModelEncodeServicer_to_server(ModelEncodeService(net), server)
    server.add_insecure_port('[::]:8888')
    server.start()
    print("Server started...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
