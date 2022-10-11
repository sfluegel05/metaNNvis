import tensorflow as tf

from src.metannvis.frameworks.framework_keys import TENSORFLOW2
from src.metannvis.frameworks.Framework import Framework


class TensorFlow2Framework(Framework):

    @staticmethod
    def get_framework_key():
        return TENSORFLOW2

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, tf.keras.Model)
        # open question: which 'class' is SavedModel?
