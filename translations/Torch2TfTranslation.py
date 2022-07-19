import os

from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

from onnx2keras import main, onnx2keras


class Torch2TfTranslation(Translation):

    @staticmethod
    def get_input():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def get_output():
        return TensorFlow2Framework.get_framework_key()

    @staticmethod
    def translate_model(model, **kwargs):
        if not PyTorchFramework.is_framework_model(model):
            raise Exception(
                f'A torch-model has to be an instance of torch.nn.Module. The given model is of type {type}')
        # TODO: delete temporary model save files
        if 'dummy_input' not in kwargs:
            raise Exception('The translation from PyTorch to Tensorflow requires providing an argument \'dummy_input\'')
        onnx_path = 'temp_torch2onnx.onnx'
        torch.onnx.export(model, kwargs['dummy_input'], onnx_path)

        # Load the ONNX file
        onnx_model = onnx.load(onnx_path)

        # Import the ONNX model to Tensorflow
        # tf_rep = prepare(onnx_model, logging_level='WARNING')
        # print(tf_rep._tf_module)
        # tf_rep.export_graph(os.path.join('..', 'models', 'mnist_tf.pb'))
        # tf_model1 = tf.saved_model.load(os.path.join('..', 'models', 'mnist_tf.pb'))

        # main(onnx_path, os.path.join('..', 'models', 'mnist_keras.h5'))
        # tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'mnist_keras.h5'))

        tf_model = onnx2keras(onnx_model)
        return tf_model

    @staticmethod
    def translate_data(data, **kwargs):
        # TODO: implement
        return data
