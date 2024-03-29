"""
A model encoding file, that takes a keras Sequential model, and encodes it to protobuf format.
"""

from google.protobuf.message import EncodeError
import tensorflow as tf
import os
from tensorflow_manager.proto.python.layer.layer_pb2 import Layer, SequentialModelLayers

class ProtoEncoder:
    def __init__(
        self, 
        model: tf.keras.models.Sequential,
        optim: str,
        metrics: list,
        save_name: str = "weights.h5"
        ) -> None:
        self.model = model
        self.optim = optim
        self.metrics = metrics
        self.save_name = save_name
        assert save_name.endswith(".h5"), "Save format of the file must be .h5"

    def get_activation(self, layer: tf.keras.layers.Dense) -> str:
        activations = tf.keras.activations.__dict__

        for k in activations:
            if layer.activation == activations[k]:
                return k
        print("Error: No corresponding activation found.")
        exit(1)

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
            shape.shape.extend(layer_shape[1:]) # todo: temporary hack
            flatten_layer.shapes.append(shape)
            layer_manager.flattenLayers.append(flatten_layer)

        elif isinstance(layer, tf.keras.layers.Dense):
            layer_type.type.append(Layer.LayerType.DENSE)
            dense_layer = layer_manager.DenseLayer()
            dense_layer.units.append(layer.units)
            activation = self.get_activation(layer)
            dense_layer.activations.append(activation)
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
        print(f"Saving model in {self.save_name}...")
        self.model.save_weights(self.save_name)
        try:
            with open(self.save_name, "rb") as fd:
                weight_details = fd.read()
        except Exception as e:
            print(f"Error occurred when reading weights: {e}")
        finally:
            os.remove(self.save_name)
        return weight_details
        

