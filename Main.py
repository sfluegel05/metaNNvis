import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import logging

import toolsets.toolset_keys
from toolsets.Captum import Captum
from translations.Torch2TfTranslation import Torch2TfTranslation
from translations.Tf2TorchTranslation import Tf2TorchTranslation
from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
from frameworks.Framework import Framework

TRANSLATIONS = [Torch2TfTranslation, Tf2TorchTranslation]
FRAMEWORKS = [PyTorchFramework, TensorFlow2Framework]
TOOLSETS = [Captum]


def translate(model, to_framework, **kwargs):
    input_key = ''
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                input_key = fw.get_framework_key()

    if input_key == '':
        raise Exception('Could not detect the model framework')

    # base case: no translation needed
    if input_key == to_framework:
        return model

    for trans in TRANSLATIONS:
        if isinstance(trans(), Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                return trans.translate(model, **kwargs)

    raise Exception(f'Could not find a translation from {input_key} to {to_framework}')

def execute(model, method_key, toolset=None, init_args=None, exec_args=None, **kwargs):
    methods = []
    if toolset is None:
        for t in TOOLSETS:
            for m in t.get_methods():
                if m.get_method_key() == method_key:
                    methods.append((m, t))
        if len(methods) == 0:
            ex_str = f'Could not find a method with key "{method_key}". The following methods are available: '
            for t in TOOLSETS:
                ex_str += f'\n\tFrom toolset {t.get_toolset_key()}: {",".join([m.get_method_key() for m in t.get_methods()])}'
            raise Exception(ex_str)

    else:
        for t in TOOLSETS:
            if t.get_toolset_key() == toolset:
                for m in t.get_methods():
                    if m.get_method_key() == method_key:
                        methods.append((m, t))
                if len(methods) == 0:
                    raise Exception(f'Could not find a method with key "{method_key}" in toolset {toolset}.'
                      + f'Available methods are: {",".join([m.get_method_key() for m in t.get_methods()])}')

        if len(methods) == 0:
            raise Exception(f'Could not find the toolset "{toolset}". Available toolsets are:'
                  f' {",".join([t.get_toolset_key() for t in TOOLSETS])}')

    model_framework = ''
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                model_framework = fw.get_framework_key()

    if model_framework == '':
        raise Exception(f'Could not detect the model framework. Available frameworks are: '
              f'{",".join([f.get_framework_key() for f in FRAMEWORKS])}')

    method, method_toolset = methods[0]
    # if multiple methods match: pick one that matches the toolset if available, pick the first registered otherwise
    if len(methods) > 1:
        if toolset is None:
            logging.warning(f'Multiple methods found for key {method_key}: ')
        else:
            logging.warning(f'Multiple methods found for key {method_key} in toolset {toolset}: ')
        logging.warning(f'{",".join([f"{m.get_method_key()} ({t.get_toolset_key()})" for m, t in methods])}')

        framework_methods = list(filter(lambda x: x[1].get_framework() == model_framework, methods))
        if len(framework_methods) > 0:
            method, method_toolset = framework_methods[0]
        logging.warning(f'Chose method {method.get_method_key() (method_toolset.get_toolset_key())}')

    # check arguments
    for key in method.get_required_init_keys():
        if key not in init_args.keys():
            raise Exception(f'The method {method.get_method_key()} requires that you include an argument \'{key}\' in'
                            f' the init_args dictionary.')
    for key in method.get_required_exec_keys():
        if key not in exec_args.keys():
            raise Exception(f'The method {method.get_method_key()} requires that you include an argument \'{key}\' in'
                            f' the exec_args dictionary.')

    model = translate(model, method_toolset.get_framework(), **kwargs)

    return method.execute(model, init_args, exec_args)


if __name__ == "__main__":
    import tensorflow as tf
    import os
    from torchvision import datasets

    tf_model = tf.keras.models.load_model(os.path.join('models', 'tf_basic_cnn_mnist'))
    mnist_test_data = datasets.FashionMNIST(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor()
    )
    mnist_test_dataloader = DataLoader(mnist_test_data, batch_size=64, shuffle=True)
    test_input_tensor, test_labels = next(iter(mnist_test_dataloader))
    test_input_tensor.requires_grad_()

    method = CAPTUM
    mnist_test_dataloader = DataLoader(mnist_test_data, batch_size=64, shuffle=True)
    execute(tf_model, 'integrated_gradients', init_args={'multiply_by_inputs': False},
            exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})


    #print(execute(tf_model, 'integrated_gradients', toolset='captum'))
    #print(execute(tf_model, 'gradiated_integers'))
    #print(execute(tf_model, 'integrated_gradients', toolset='tf-keras-vis'))  # todo: warning + use correct toolset
    # todo: exceptions

