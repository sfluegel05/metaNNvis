import torch

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


def translate(model, to_framework, *args, **kwargs):
    input_key = ''
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                input_key = fw.get_framework_key()

    if input_key == '':
        print('Could not detect the model framework')
        return False

    # base case: no translation needed
    if input_key == to_framework:
        return model

    for trans in TRANSLATIONS:
        if isinstance(trans(), Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                return trans.translate(model, *args, **kwargs)

    print(f'Could not find a translation from {input_key} to {to_framework}')
    return False


def execute(model, method_key, toolset=None, *args, **kwargs):
    methods = []
    if toolset is None:
        for t in TOOLSETS:
            for m in t.get_methods():
                if m.get_method_key() == method_key:
                    methods.append((m, t))
        if len(methods) == 0:
            print(f'Could not find a method with key "{method_key}". The following methods are available: ')
            for t in TOOLSETS:
                print(f'\tFrom toolset {t.get_toolset_key()}: {",".join([m.get_method_key() for m in t.get_methods()])}')
            return False

    else:
        for t in TOOLSETS:
            if t.get_toolset_key() == toolset:
                for m in t.get_methods():
                    if m.get_method_key() == method_key:
                        methods.append((m, t))
                if len(methods) == 0:
                    print(f'Could not find a method with key "{method_key}" in toolset {toolset}.', end=' ')
                    print(f'Available methods are: {",".join([m.get_method_key() for m in t.get_methods()])}')
                    return False

        if len(methods) == 0:
            print(f'Could not find the toolset "{toolset}". Available toolsets are:'
                  f' {",".join([t.get_toolset_key() for t in TOOLSETS])}')

    model_framework = ''
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                model_framework = fw.get_framework_key()

    if model_framework == '':
        print(f'Could not detect the model framework. Available frameworks are: '
              f'{",".join([f.get_framework_key() for f in FRAMEWORKS])}')
        return False

    method, method_toolset = methods[0]
    if len(methods) > 1:
        if toolset is None:
            print(f'Multiple methods found for key {method_key}: ', end='')
        else:
            print(f'Multiple methods found for key {method_key} in toolset {toolset}: ', end='')
        print(f'{",".join([f"{m.get_method_key()} ({t.get_toolset_key()})" for m, t in methods])}')

        framework_methods = list(filter(lambda x: x[1].get_framework() == model_framework, methods))
        if len(framework_methods) > 0:
            method, method_toolset = framework_methods[0]
        print(f'Chose method {method.get_method_key() (method_toolset.get_toolset_key())}')

    model = translate(model, method_toolset.get_framework(), *args, **kwargs)

    return method.execute(model, *args, **kwargs)


if __name__ == "__main__":
    import tensorflow as tf
    import os
    tf_model = tf.keras.models.load_model(os.path.join('models', 'tf_basic_cnn_mnist'))
    print(execute(tf_model, 'integrated_gradients', input=torch.rand(1, 1, 28, 28), target=1))
    #print(execute(tf_model, 'integrated_gradients', toolset='captum'))
    #print(execute(tf_model, 'gradiated_integers'))
    #print(execute(tf_model, 'integrated_gradients', toolset='tf-keras-vis'))  # todo: warning + use correct toolset
    # todo: exceptions

