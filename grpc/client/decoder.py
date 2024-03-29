"""
A file that houses the decoder that takes an encoded Tensorflow Keras model, and decodes it, 
and reconstructs it.
"""
from google.protobuf.message import DecodeError
import tensorflow as tf
import os
from tensorflow_manager.proto.python.layer.layer_pb2 import SequentialModelLayers, Layer

class ProtoDecoder:
    """
    Class involved with decoding a given proto model, and reconstructing it into a Keras model
    """
    def __init__(self) -> None:
        pass

    def decode_model(self, model_str: bytes) -> tf.keras.models.Sequential:
        """
        Decodes a given model string, and restructures it as a Keras model.
        Currently only supports Sequential model.

        Args:
            model_str (bytes): The bytestring that encodes the layer details. 

        Returns:
            tf.keras.models.Sequential: The reconstructed Keras model
        """
        model_layers = []
        encoded_layers = SequentialModelLayers()
        encoded_layers.ParseFromString(model_str)
        layer_counter = 0

        for layer in encoded_layers.layers:
            layer_counter += 1
            layer_type = layer.type.pop(0)

            if layer_type == Layer.LayerType.FLATTEN:
                flatten_layer = encoded_layers.flattenLayers.pop(0)
                shape = flatten_layer.shapes.pop(0).shape
                decoded_layer = tf.keras.layers.Flatten(input_shape=tuple(shape))

            elif layer_type == Layer.LayerType.DENSE:
                dense_layer = encoded_layers.denseLayers.pop(0)
                units = dense_layer.units.pop(0)
                activation = dense_layer.activations.pop(0)
                names = dense_layer.names.pop(0)
                decoded_layer = tf.keras.layers.Dense(units, activation=activation, name=names)

            elif layer_type == Layer.LayerType.DROPOUT:
                dropout_layer = encoded_layers.dropoutLayers.pop(0)
                rate = dropout_layer.ratio.pop(0)
                decoded_layer = tf.keras.layers.Dropout(rate)
                
            else:
                print(f"Error: Encountered layer without available conversion {layer_type}. Please consult layers.proto file. Exiting.")
                raise DecodeError

            model_layers.append(decoded_layer)
        
        return tf.keras.models.Sequential(model_layers)
    
    def load_weights(
        self, 
        filename: str, 
        model: tf.keras.models.Sequential, 
        weight_str: bytes
        ) -> tf.keras.models.Sequential:
        """
        Loads the weights to the decoded model after it has been reconstituted

        Args:
            filename (str): filename of the .hd5 weights file
            model (tf.keras.models.Sequential): the reconstituted model, received over gRPC network
            weight_str (bytes): the bytestring that includes the encoded weights for the model

        Returns:
            tf.keras.models.Sequential: model with the loaded weights
        """
        assert filename.endswith(".h5"), "Invalid save file extension. Must be a .h5 file."
        try:
            with open(filename, "wb") as fd:
                fd.write(weight_str)
        except Exception: 
            print(f"Error writing the weight binaries to file.")

        try: 
            model.load_weights(filename)
        except Exception as e:
            print(f"Error loading model weights: {e}")
        finally:
            os.remove(filename)
        return model
        
