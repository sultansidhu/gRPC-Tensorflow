"""
A model encoding file, that takes a keras Sequential model, and encodes it to protobuf format.
"""

from google.protobuf.message import EncodeError
import tensorflow as tf
import os
import shutil

from tensorflow_manager.proto.python.layer.layer_pb2 import Layer, SequentialModelLayers
from tensorflow_manager.model.model import ModelGenerator

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
    
    def compile_model(self):
        """
        This function compiles the model, and returns it.
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer=self.optim,
            loss=loss_fn,
            metrics=self.metrics
        )

    def encode_entire_model(self) -> bytes:
        """
        Function for encoding the entire model, layer by layer, into Protobuf format.

        Returns:
            bytes: protobuf encoded model
        """
        #self.compile_model() # uncomment when compiling own model
        self.model.save(self.save_name)
        shutil.make_archive(self.save_name, "zip", self.save_name)
        with open(f"{self.save_name}.zip", "rb") as fd:
            encoded_model = fd.read()
        shutil.rmtree(self.save_name) # remove tree of the saved model file
        os.remove(f"{self.save_name}.zip")
        return encoded_model

if __name__ == "__main__":
    model = ModelGenerator().get_model()
    encoder = ProtoEncoder(model, "adam", ["accuracy"])
    print(encoder.encode_entire_model())[:100]
