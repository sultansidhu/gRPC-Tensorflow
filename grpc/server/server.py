"""
The main file for serving the encoded model using the model encoder.
"""

# module imports
import grpc
from concurrent import futures
import time

# import Keras model generator
from tensorflow_manager.model.model import ModelGenerator

# import tensorflow
import tensorflow as tf

# import protobuf created files
import tensorflow_manager.proto.python.service.service_pb2 as pb2
import tensorflow_manager.proto.python.service.service_pb2_grpc as pb2_grpc

# importing encoder to encode model, and send
from encoder import ProtoEncoder


class ModelEncodeService(pb2_grpc.ModelEncodeServicer):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.encoder = ProtoEncoder(self.model, "adam", "accuracy")

    def GetEncodedModel(self, request, context):
        start_time = time.time()
        encoded_model = self.encoder.encode_model()
        weights = self.encoder.encode_weights()
        hyperparams = pb2.HyperParams()
        hyperparams.loss = pb2.HyperParams.LossFunction.SparseCategoricalCE
        hyperparams.fromLogits = True
        filename = self.encoder.save_name
        result = {
            'model': encoded_model, 
            'weights': weights, 
            'hyperparams': hyperparams, 
            'fileName': filename
        }
        end_time = time.time()
        print(f"Time taken for returning response - {end_time - start_time}")
        return pb2.ModelResponse(**result)


def serve():
    net = ModelGenerator().get_model()
    net.load_weights("weights.h5")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    net.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

    net.evaluate(x_test,  y_test, verbose=2)


    print(net.summary())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ModelEncodeServicer_to_server(ModelEncodeService(net), server)
    server.add_insecure_port('[::]:8888')
    server.start()
    print("Server started...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
