"""
A python file that keeps the definition of a simple tensorflow model.
"""
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class ModelGenerator:
    """
    This class is a generator class, used to generate various kinds of models.
    Currently supported array of models:
    1. MNIST model (Keras sample)
    -- LIST TO BE CONTINUED --
    """
    def __init__(self) -> None:
        pass
    
    def get_model(self, layer_list: list = []) -> tf.keras.models.Sequential:
        """
        A model generator function that takes a list of layers, and generates
        a keras Sequential model.

        Args:
            layer_list (list): List of tf.keras.layers objects, which is used to
            instantiate tf.keras Sequential models

        Returns:
            tf.keras.models.Sequential: a model constituted of the layers.
        """
        model = None
        if not layer_list:
            layer_list = [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10)
            ]
        if self.type == "mnist":
            model = tf.keras.models.Sequential(layer_list)
        if model:
            return model
        else:
            print("Error: Model type is not supported!")
            exit(1)
        

model = ModelGenerator().get_model()

predictions = model(x_train[:1]).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print(model)
print(dir(model))