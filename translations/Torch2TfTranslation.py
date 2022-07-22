from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
import torch
import onnx

from onnx2keras import onnx2keras


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

        # deactivate dropout
        model.eval()

        onnx_path = 'temp_torch2onnx.onnx'
        torch.onnx.export(model, kwargs['dummy_input'], onnx_path)

        # Load the ONNX file
        onnx_model = onnx.load(onnx_path)

        tf_model = onnx2keras(onnx_model)
        return tf_model

    @staticmethod
    def translate_data(data, **kwargs):
        if isinstance(data, list):
            return list(map(lambda x: Torch2TfTranslation.translate_data(x), data))
        elif isinstance(data, torch.Tensor):
            data = data.detach().numpy()
            # move channel dimension to the last position for 2d images
            if len(data.shape) == 4:
                return data.reshape((data.shape[0], data.shape[2], data.shape[3], data.shape[1]))
            else:
                # assume the tensor is an input tensor and reshape
                shape = kwargs['model'].inputs[0].shape.as_list()
                shape = [-1 if val is None else val for val in shape]
                return data.detach().numpy().reshape(shape)

        return data
