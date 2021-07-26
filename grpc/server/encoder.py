"""
A model encoding file, that takes a keras Sequential model, and encodes it to protobuf format.
"""

from google.protobuf.message import EncodeError
import tensorflow as tf
import os
import shutil
from tensorflow_manager.proto.python.layer.layer_pb2 import Layer, SequentialModelLayers

class ProtoEncoder:
    def __init__(
        self, 
        model: tf.keras.models.Sequential,
        optim: str,
        metrics: list,
        save_name: str = "model"
        ) -> None:
        self.model = model
        self.optim = optim
        self.metrics = metrics
        self.save_name = save_name
        assert not save_name.endswith(".h5"), "Save format of the file must not be .h5"

    def encode_entire_model(self) -> bytes:
        """
        Function for encoding the entire model, layer by layer, into Protobuf format.

        Returns:
            bytes: protobuf encoded model
        """
        self.model.save(self.save_name)
        shutil.make_archive(self.save_name, "zip", self.save_name)
        with open(f"{self.save_name}.zip", "rb") as fd:
            encoded_model = fd.read()
        shutil.rmtree(self.save_name) # remove tree of the saved model file
        os.remove(f"{self.save_name}.zip")
        return encoded_model

    def encode_weights(self) -> bytes:
        """
        A function that encodes the weights of the model into a protobuf format.

        Returns:
            bytes: Returns a bytestring that encodes the model weights
        """
        print(f"Saving model in {self.save_name}...")
        self.model.save_weights(self.save_name)
        try:
            with open(f"{self.save_name}.h5", "wb") as fd:
                fd.write("haha")
            with open(f"{self.save_name}.h5", "rb") as fd:
                weight_details = fd.read()
        except Exception as e:
            print(f"Error occurred when reading weights: {e}")
        finally:
            os.remove(f"{self.save_name}.h5")
        return weight_details
        

