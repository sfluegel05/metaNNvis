import logging

import seaborn

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.AbstractFeatureVisualizationMethod import AbstractFeatureVisualizationMethod
from src.metannvis.toolsets.Captum import Captum
from src.metannvis.toolsets.TfKerasVis import TfKerasVis
from src.metannvis.translations.Torch2TfTranslation import Torch2TfTranslation
from src.metannvis.translations.Tf2TorchTranslation import Tf2TorchTranslation
from src.metannvis.frameworks.PyTorchFramework import PyTorchFramework
from src.metannvis.frameworks.TensorFlow2Framework import TensorFlow2Framework
from src.metannvis.translations.Translation import Translation
from src.metannvis.frameworks.Framework import Framework

TRANSLATIONS = [Torch2TfTranslation, Tf2TorchTranslation]
FRAMEWORKS = [PyTorchFramework, TensorFlow2Framework]
TOOLSETS = [Captum, TfKerasVis]


def detect_model_framework(model):
    """Detects which framework a neural network belongs to.

    Parameters
    ----------
    model : any
        the neural network

    Returns
    -------
    str
        the model's framework key

    Raises
    ------
    Exception
        if the model matches none of the available frameworks
    """
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                return fw.get_framework_key()

    raise Exception('Could not detect the model framework')


def translate_model(model, to_framework, **kwargs):
    """Translates a neural network from one framework into another.

    Parameters
    ----------
    model : any
        the neural network
    to_framework : str
        the framework key of the target framework
    **kwargs : dict
        additional arguments required for the translation

    Returns
    -------
    translated model : any
        a neural network in the target framework which behaves (nearly) equivalent to the input model

    Raises
    ------
    Exception
        if no translation between the required frameworks is available
    """
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


def translate_data(data_dict, to_framework, original_model, translated_model, translate_to_numpy=False):
    """Translate input / output data from one framework to another.

    Parameters
    ----------
    data_dict : {dict, array-like}
        the data to be translated, if it is a dict, each value is translated individually
    to_framework : str
        the key of the target framework
    original_model : any
        the model before the model translation (used to determine which framework the data belongs to)
    translated_model : any
        the model after the model translation (used to fit the data shape to the model's required input shape)
    translate_to_numpy : bool, optional
        if True, don't translate to the to_framework, but to a numpy array, default: False

    Returns
    -------
    {dict, array-like}
        the translation of data_dict

    Raises
    ------
    Exception
        if no translation between the model's framework and the target framework can be found

    """
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
                        return_dict[key] = trans.translate_data(value, model=translated_model, key=key,
                                                                translate_to_numpy=translate_to_numpy)
                    return return_dict
                else:
                    return trans.translate_data(data_dict, model=translated_model,
                                                translate_to_numpy=translate_to_numpy)

    raise Exception(f'Could not find a translation from {input_key} to {to_framework}')


def perform_attribution(model, method_key, toolset=None, init_args=None, exec_args=None, plot=False, **kwargs):
    """Finds and executes an attribution method that fits the method_key, translates model and input data if needed.

    Parameters
    ----------
    model : any
        the neural network on which to perform the attribution
    method_key : str
        the attribution method to perform
    toolset : str, optional
        the toolset from which the attribution method should be selected. If the given toolset doesn't contain the
        given method, another toolset is chosen. If no toolset is given, the toolset which doesn't require a translation
        is prioritized.
    init_args : dict, optional
        the arguments to initialize the attribution method
    exec_args : dict, optional
        the arguments to execute the attribution method
    plot : bool, optional
        if True, plot the attribution results as a heat map
    **kwargs : dict
        additional arguments for the translation (e.g., a 'dummy input' for the Torch2TfTranslation)

    Returns
    -------
    res : numpy.ndarray
        the output of the attribution method, translated to a numpy array
    """
    return execute(model, method_key, toolset, init_args, exec_args, plot,
                   method_type=AbstractAttributionMethod.get_method_type(), **kwargs)


def perform_feature_visualization(model, method_key, toolset=None, init_args=None, exec_args=None, plot=False,
                                  **kwargs):
    """Finds and executes a feature visualization method that fits the method_key, translates model and input data if
    needed.

    Parameters
    ----------
    model : any
        the neural network on which to perform the feature visualization
    method_key : str
        the feature visualization method to perform
    toolset : str, optional
        the toolset from which the feature visualization method should be selected. If the given toolset doesn't contain
        the given method, another toolset is chosen. If no toolset is given, the toolset which doesn't require a
        translation is prioritized.
    init_args : dict, optional
        the arguments to initialize the feature visualization method
    exec_args : dict, optional
        the arguments to execute the feature visualization method
    plot : bool, optional
        if True, plot the feature visualization results as a heat map
    **kwargs : dict
        additional arguments for the translation (e.g., a 'dummy input' for the Torch2TfTranslation)

    Returns
    -------
    res : numpy.ndarray
        the output of the feature visualization method, translated to a numpy array
    """
    return execute(model, method_key, toolset, init_args, exec_args, plot,
                   method_type=AbstractFeatureVisualizationMethod.get_method_type(), **kwargs)


def execute(model, method_key, toolset=None, init_args=None, exec_args=None, plot=False,
            method_type=AbstractAttributionMethod.get_method_type(), **kwargs):
    """Finds and executes an introspection method that fits the method_key, translates model and input data if needed.

    Note that this method is not supposed to be called directly, but is just a helper method for perform_attribution()
    and perform_feature_visualization().

    Parameters
    ----------
    model : any
        the neural network on which to perform the introspection
    method_key : str
        the introspection method to perform
    toolset : str, optional
        the toolset from which the introspection method should be selected. If the given toolset doesn't contain the
        given method, another toolset is chosen. If no toolset is given, the toolset which doesn't require a translation
        is prioritized.
    init_args : dict, optional
        the arguments to initialize the introspection method
    exec_args : dict, optional
        the arguments to execute the introspection method
    plot : bool, optional
        if True, plot the introspection results as a heat map
    method_type : {'attribution', 'feature_visualization'}
        the method type. Only methods that have the correct type are considered.
    **kwargs : dict
        additional arguments for the translation (e.g., a 'dummy input' for the Torch2TfTranslation)

    Returns
    -------
    res : {numpy.ndarray, dict}
        if the method is a Captum layer method and the init_args don't contain a 'layer' key, a dict with the
        intermediate translation results is returned, otherwise it is the output of the introspection method,
        translated to a numpy array

    Raises
    ------
    Exception
        if there is no method which fits the method_key, if the model doesn't belong to one of the known frameworks or
        if required init_args / exec_args are missing
    """
    if init_args is None:
        init_args = {}
    if exec_args is None:
        exec_args = {}
    method_key = method_key.lower()
    methods = []
    if toolset is not None:
        if toolset not in [t.get_toolset_key() for t in TOOLSETS]:
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
        logging.warning(f'Chose method {method.get_method_key()}({method_toolset.get_toolset_key()})')

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
                'translated_exec_args': translated_exec_args, 'plot': plot}

    res = method.execute(translated_model, translated_init_args, translated_exec_args)

    if plot:
        plot_results(res)

    res = translate_data(res, detect_model_framework(model), translated_model, model, translate_to_numpy=True)

    return res


def finish_execution_with_layer(intermediate_output, layer_key):
    """Rerun execute with an additional layer argument that depends on the model translation.

    Parameters
    ----------
    intermediate_output : dict
        the output of perform_attribution(), containing the method to be executed and the translated model and arguments
    layer_key : str
        the key of the model layer for which the method should be executed

    Returns
    -------
    numpy.ndarray
        the method's output, translated to numpy

    """
    model = intermediate_output['translated_model']
    init_args = intermediate_output['translated_init_args']
    init_args['layer'] = getattr(model, layer_key)

    res = intermediate_output['method'].execute(model, init_args, intermediate_output['translated_exec_args'])

    if intermediate_output['plot']:
        plot_results(res)

    return res


def plot_results(attr):
    """Plots the introspection results.

    The first dimension is assumed to be the batch-dimension, each row in the plot corresponds to one element in the
    batch. If attr is 4-dimensional, the second dimension (or the forth, if the second and third are equal) is used as
    the channel dimension with one column per channel. The plot is displayed and saved in a time-stamped file.

    Parameters
    ----------
    attr : array-like
        The output of some introspection method

    """
    import matplotlib.pyplot as plt
    import datetime
    import numpy as np

    if not isinstance(attr, np.ndarray):
        attr = attr.detach().numpy()

    n_cols = attr.shape[0]
    if len(attr.shape) == 4 and attr.shape[1] == attr.shape[2]:
        attr = np.reshape(attr, (n_cols, attr.shape[3], attr.shape[1], attr.shape[2]))
    if len(attr.shape) == 4:
        figure = plt.figure(figsize=(5 * n_cols, 5 * attr.shape[1]))
    else:
        figure = plt.figure(figsize=(5 * n_cols, 5))

    counter = 0
    for i in range(n_cols):
        if len(attr.shape) == 4:
            for c in range(attr.shape[1]):
                figure.add_subplot(attr.shape[1], n_cols, counter + 1 + c * n_cols)
                plt.title(f'Channel {c}')
                seaborn.heatmap(attr[i][c].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                                center=0, xticklabels=5, yticklabels=5)
            counter += 1
        else:
            figure.add_subplot(1, n_cols, counter + 1)
            counter += 1
            seaborn.heatmap(attr[i].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                            center=0, xticklabels=5, yticklabels=5)
    plt.savefig(f"res_plot{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", bbox_inches='tight')
    plt.show()
