from frameworks.Framework import Framework
import tensorflow as tf


class TensorFlow2Framework(Framework):

    @staticmethod
    def get_framework_key():
        return 'tf2'

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, tf.keras.Model)
        # open question: which 'class' is SavedModel?
