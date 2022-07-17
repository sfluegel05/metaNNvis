import os.path

import logging

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.AbstractFeatureVisualizationMethod import AbstractFeatureVisualizationMethod
from toolsets.Captum import Captum
from toolsets.TfKerasVis import TfKerasVis
from translations.Torch2TfTranslation import Torch2TfTranslation
from translations.Tf2TorchTranslation import Tf2TorchTranslation
from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
from frameworks.Framework import Framework

TRANSLATIONS = [Torch2TfTranslation, Tf2TorchTranslation]
FRAMEWORKS = [PyTorchFramework, TensorFlow2Framework]
TOOLSETS = [Captum, TfKerasVis]


def detect_model_framework(model):
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                return fw.get_framework_key()

    raise Exception('Could not detect the model framework')


def translate_model(model, to_framework, **kwargs):
    to_framework = to_framework.lower()
    input_key = detect_model_framework(model)

    # base case: no translation needed
    if input_key == to_framework:
        return model

    for trans in TRANSLATIONS:
        if isinstance(trans(), Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                return trans.translate_model(model, **kwargs)

    raise Exception(f'Could not find a translation from {input_key} to {to_framework}')


def translate_data(data_dict, to_framework, original_model, translated_model):
    input_key = detect_model_framework(original_model)

    # base case: no translation needed
    if input_key == to_framework:
        return data_dict

    for trans in TRANSLATIONS:
        if isinstance(trans(), Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                if isinstance(data_dict, dict):
                    return_dict = {}
                    for key, value in data_dict.items():
                        return_dict[key] = trans.translate_data(value, model=translated_model, key=key)
                    return return_dict
                else:
                    return trans.translate_data(data_dict, model=translated_model)

    raise Exception(f'Could not find a translation from {input_key} to {to_framework}')


def perform_attribution(model, method_key, toolset=None, init_args=None, exec_args=None, **kwargs):
    return execute(model, method_key, toolset, init_args, exec_args,
                   method_type=AbstractAttributionMethod.get_method_type(), **kwargs)


def perform_feature_visualization(model, method_key, toolset=None, init_args=None, exec_args=None, **kwargs):
    return execute(model, method_key, toolset, init_args, exec_args,
                   method_type=AbstractFeatureVisualizationMethod.get_method_type(), **kwargs)


def execute(model, method_key, toolset=None, init_args=None, exec_args=None,
            method_type=AbstractAttributionMethod.get_method_type(), **kwargs):
    method_key = method_key.lower()
    methods = []
    if toolset is not None:
        if toolset not in TOOLSETS:
            logging.warning(f'Could not find the toolset "{toolset}". Available toolsets are:'
                            f' {",".join([t.get_toolset_key() for t in TOOLSETS])}')
        else:
            for t in TOOLSETS:
                if t.get_toolset_key() == toolset:
                    for m in t.get_methods(method_type):
                        if m.get_method_key() == method_key:
                            methods.append((m, t))
                    if len(methods) == 0:
                        logging.warning(f'Could not find a method with key "{method_key}" in toolset {toolset}.'
                                        + f'Available methods are: {",".join([m.get_method_key() for m in t.get_methods(method_type)])}')
        if len(methods) == 0:
            logging.info(f'Looking for method "{method_key}" in all toolsets')
            toolset = None
    if toolset is None:
        for t in TOOLSETS:
            for m in t.get_methods(method_type):
                if m.get_method_key() == method_key:
                    methods.append((m, t))
        if len(methods) == 0:
            ex_str = f'Could not find a method with key "{method_key}". The following methods are available: '
            for t in TOOLSETS:
                ex_str += f'\n\tFrom toolset {t.get_toolset_key()}: ' \
                          f'{",".join([m.get_method_key() for m in t.get_methods(method_type)])}'
            raise Exception(ex_str)

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
        logging.warning(f'Chose method {method.get_method_key()(method_toolset.get_toolset_key())}')

    # check arguments
    for key in method.get_required_init_keys():
        if key not in init_args.keys() and key != 'layer':
            raise Exception(f'The method {method.get_method_key()} requires that you include an argument \'{key}\' in'
                            f' the init_args dictionary.')
    for key in method.get_required_exec_keys():
        if key not in exec_args.keys():
            raise Exception(f'The method {method.get_method_key()} requires that you include an argument \'{key}\' in'
                            f' the exec_args dictionary.')

    translated_model = translate_model(model, method_toolset.get_framework(), **kwargs)
    translated_init_args = translate_data(init_args, method_toolset.get_framework(), model, translated_model)
    translated_exec_args = translate_data(exec_args, method_toolset.get_framework(), model, translated_model)
    # special case: if the method requires a 'layer' argument, the layer has to be a layer from the target framework
    # model. Since this can be neither specified by the user beforehand nor determined automatically, ask the user at
    # this point to select a layer
    if "layer" in method.get_required_init_keys() and "layer" not in init_args \
            and method_toolset.get_framework() == PyTorchFramework.get_framework_key():
        logging.warning(f'The method you want to call requires you to provide a PyTorch model layer. Please call '
                        f'\'finish_execution_with_layer(intermediate_output, layer_key)\' with the return value of this'
                        f' call as \'intermediate_output\' and one of '
                        f'{[elem for elem in dict(translated_model.named_children()).keys()]}'
                        f' as the \'layer_key\'')
        return {'method': method, 'translated_model': translated_model, 'translated_init_args': translated_init_args,
                'translated_exec_args': translated_exec_args}

    return method.execute(translated_model, translated_init_args, translated_exec_args)


def finish_execution_with_layer(intermediate_output, layer_key):
    model = intermediate_output['translated_model']
    init_args = intermediate_output['translated_init_args']
    init_args['layer'] = getattr(model, layer_key)
    return intermediate_output['method'].execute(model, init_args, intermediate_output['translated_exec_args'])


if __name__ == "__main__":
    import tensorflow as tf

    new_model = tf.keras.models.load_model(os.path.join('translations', 'models', 'mnist_tf.pb'))
    print(type(new_model))
    print(new_model)

    # print(execute(tf_model, 'integrated_gradients', toolset='captum'))
    # print(execute(tf_model, 'gradiated_integers'))
    # print(execute(tf_model, 'integrated_gradients', toolset='tf-keras-vis')) # should put out a warning and use the correct toolset
