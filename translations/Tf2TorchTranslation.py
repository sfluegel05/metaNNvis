import os

import torch
import numpy as np

from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
import tensorflow as tf
from onnx2torch.converter import convert


class Tf2TorchTranslation(Translation):

    @staticmethod
    def get_input():
        return TensorFlow2Framework.get_framework_key()

    @staticmethod
    def get_output():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def translate_model(model, **kwargs):
        # TODO: clean up console output
        tf_path = os.path.join('models', 'temp_tf')
        tf.saved_model.save(model, tf_path)
        os.system('python3 -m tf2onnx.convert --saved-model ./models/temp_tf --output ./models/temp_tf2onnx.onnx')

        onnx_path = os.path.join('models', 'temp_tf2onnx.onnx')
        torch_model = convert(onnx_path)

        return torch_model

    @staticmethod
    def translate_data(data, **kwargs):
        # translate lists recursively
        if isinstance(data, list):
            return list(map(lambda x: Tf2TorchTranslation.translate_data(x), data))
        elif isinstance(data, str) and 'model' in kwargs and 'key' in kwargs and kwargs['key'] == 'layer':
            if data not in [elem for elem in dict(kwargs['model'].named_children()).keys()]:
                raise Exception(f'The value provided for the argument \'layer\' (\'{data}\') does not reference a '
                                f'layer in the translated model. Valid options are: '
                                f'{[elem for elem in dict(kwargs["model"].named_children()).keys()]}')
            return getattr(kwargs['model'], data)
        # convert torch to torch array
        elif isinstance(data, tf.Tensor):
            try:
                data = data.detach().numpy()
            except AttributeError:
                data = data.numpy()
            if np.issubdtype(data.dtype, np.double):
                data = data.astype(np.single)
            elif np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.int_)
            if 'translate_to_numpy' in kwargs and kwargs['translate_to_numpy']:
                return data
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.double):
                data = data.astype(np.single)
            elif np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.int64)
            if 'translate_to_numpy' in kwargs and kwargs['translate_to_numpy']:
                return data
            return torch.from_numpy(data)
        else:
            return data


if __name__ == '__main__':
    model = tf.saved_model.load(os.path.join('..', 'project_preparation_demo', 'models', 'mnist_tf_pretrained'))
    print(model)
    torch_model = Tf2TorchTranslation.translate_model(model)
    print(torch_model)
    input = torch.randn([1, 1, 28, 28])
    print(torch_model(input))
