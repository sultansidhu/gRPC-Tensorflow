"""
A model encoding file, that takes a keras Sequential model, and encodes it to protobuf format.
"""

from google.protobuf.message import EncodeError
import tensorflow as tf
from tensorflow_manager.proto.python.layer.layer_pb2 import Layer, SequentialModelLayers

class ProtoEncoder:
    def __init__(
        self, 
        model: tf.keras.models.Sequential,
        optim: str,
        metrics: list
        ) -> None:
        self.model = model
        self.optim = optim
        self.metrics = metrics

    def encode_layer(self, layer, layer_manager) -> Layer:
        """
        Function for encoding an individual layer within the Tensorflow Keras model.

        Args:
            layer (tf.keras.layers): Individual layer in the tensorflow model
            layer_manager (SequentialModelLayers): Manager for all layers within the model.
        """
        layer_type = Layer()

        if isinstance(layer, tf.keras.layers.Flatten):
            layer_type.type.append(Layer.LayerType.FLATTEN)
            flatten_layer = layer_manager.FlattenLayer()
            shape = layer_manager.Shape()
            layer_shape = list(layer.input_shape)
            print(f"LAYER shape is {layer_shape}")
            shape.shape.extend(layer_shape[1:]) # todo: temporary hack
            flatten_layer.shapes.append(shape)
            layer_manager.flattenLayers.append(flatten_layer)

        elif isinstance(layer, tf.keras.layers.Dense):
            layer_type.type.append(Layer.LayerType.DENSE)
            dense_layer = layer_manager.DenseLayer()
            print(layer.units, "DENSE")
            dense_layer.units.append(layer.units)
            dense_layer.activations.append(layer.activation)
            # we must make the following distinction because proto
            # expects a string
            if layer.name is not None:
                dense_layer.names.append(layer.name)
            else:
                dense_layer.names.append("")
            layer_manager.denseLayers.append(dense_layer)

        elif isinstance(layer, tf.keras.layers.Dropout):
            layer_type.type.append(Layer.LayerType.DROPOUT)
            dropout_layer = layer_manager.DropoutLayer()
            print(layer.rate, "DROPOUT")
            dropout_layer.ratio.append(layer.rate)
            layer_manager.dropoutLayers.append(dropout_layer)
        
        else:
            print(f"Error: Encoding for layer {layer} does not exist. Please add or consult the proto file.")
            raise EncodeError

        return layer_type 


    def encode_model(self) -> bytes:
        """
        Function for encoding the entire model, layer by layer, into Protobuf format.

        Returns:
            bytes: protobuf encoded model
        """
        layers = SequentialModelLayers()
        for layer in self.model.layers:
            encoded_layer = self.encode_layer(layer, layers)
            layers.layers.append(encoded_layer)
        return layers.SerializeToString()
    
    def encode_weights(self) -> bytes:
        """
        A function that encodes the weights of the model into a protobuf format.

        Returns:
            bytes: Returns a bytestring that encodes the model weights
        """
        pass

