"""
The main client file for sending a Keras model over gRPC.
"""
import grpc

import tensorflow as tf

import tensorflow_manager.proto.python.service.service_pb2 as pb2
import tensorflow_manager.proto.python.service.service_pb2_grpc as pb2_grpc

from decoder import ProtoDecoder


class ModelEncodeClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = '35.184.206.225'
        self.server_port = 8888

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.ModelEncodeStub(self.channel)

    def get_url(self, message):
        """
        Client function to call the rpc for GetServerResponse
        """
        message = pb2.Request(message=message)
        response = self.stub.GetEncodedModel(message)
        encoded_model = response.model
        encoded_weights = response.weights
        decoder = ProtoDecoder()
        model = decoder.decode_model(encoded_model)
        weighted_model = decoder.load_weights(
            response.fileName,
            model,
            encoded_weights
        )
        return weighted_model



if __name__ == '__main__':
    client = ModelEncodeClient()
    decoded_model = client.get_url(ready=True) 
    print(decoded_model.summary())

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    decoded_model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

    decoded_model.evaluate(x_test,  y_test, verbose=2)