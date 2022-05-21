from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


class Torch2TfTranslation(Translation):

    @staticmethod
    def get_input():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def get_output():
        return TensorFlow2Framework.get_framework_key()

    @staticmethod
    def translate(model, *args, **kwargs):
        if not isinstance(model, torch.nn.Module):
            raise Exception(
                f'A torch-model has to be an instance of torch.nn.Module. The given model is of type {type}')


        onnx_path = 'temp_torch2onnx.onnx'
        torch.onnx.export(model, kwargs['dummy_input'], onnx_path)

        # Load the ONNX file
        onnx_model = onnx.load(onnx_path)

        # Import the ONNX model to Tensorflow
        tf_rep = prepare(onnx_model)

        tf_rep.export_graph('models/mnist_tf.pb')
        tf_model = tf.saved_model.load('models/mnist_tf.pb')

        return tf_model
