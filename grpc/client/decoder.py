"""
A file that houses the decoder that takes an encoded Tensorflow Keras model, and decodes it, 
and reconstructs it.
"""
from google.protobuf.message import DecodeError
import tensorflow as tf
import os
import shutil
from zipfile import ZipFile
from tensorflow_manager.proto.python.layer.layer_pb2 import SequentialModelLayers, Layer

class ProtoDecoder:
    """
    Class involved with decoding a given proto model, and reconstructing it into a Keras model
    """
    def __init__(self) -> None:
        pass

    def decode_model(self, model_str: bytes, name: str) -> tf.keras.models.Sequential:
        """
        Decodes a given model string, and restructures it as a Keras model.
        Currently only supports Sequential model.

        Args:
            model_str (bytes): The bytestring that encodes the layer details. 

        Returns:
            tf.keras.models.Sequential: The reconstructed Keras model
        """
        with open(f"{name}.zip", "wb") as fd:
            fd.write(model_str)
        with ZipFile(f"{name}.zip", "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(name)
        os.remove(f"{name}.zip")
        model = tf.keras.models.load_model(name)
        shutil.rmtree(name)
        return model
        
