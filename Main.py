import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import logging

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

    model = translate(model, method_toolset.get_framework(), **kwargs)

    return method.execute(model, init_args, exec_args)


if __name__ == "__main__":
    import tensorflow as tf
    import os
    import matplotlib.pyplot as plt
    from torchvision import datasets

    tf_model = tf.keras.models.load_model(os.path.join('models', 'tf_basic_cnn_mnist'))
    test_data = datasets.FashionMNIST(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    test_input_tensor, test_labels = next(iter(test_dataloader))
    test_input_tensor.requires_grad_()

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    n_rows = 5
    for i in range(n_rows):
        label = test_labels[i].item()
        print(label)
        attr = execute(tf_model, 'integrated_gradients', init_args={'multiply_by_inputs': False},
                       exec_args={'inputs': test_input_tensor, 'target': label})
        attr = attr.detach().numpy()

        img = test_input_tensor[i][0].detach()
        figure = plt.figure(figsize=(20, 20))
        figure.add_subplot(n_rows, 2, i * 2 + 1)
        plt.title(f'Label: {labels_map[label]}')
        plt.axis("off")
        plt.imshow(img, cmap="gray")
        figure.add_subplot(n_rows, 2, i * 2 + 2)
        plt.title(f'Integrated Gradients')
        plt.axis("off")
        plt.imshow(attr[0][0], cmap="gray")
        #plt.savefig(f"integrated_gradients_fashion_mnist_demo_{i}.png")

    plt.show()

    #print(execute(tf_model, 'integrated_gradients', toolset='captum'))
    #print(execute(tf_model, 'gradiated_integers'))
    #print(execute(tf_model, 'integrated_gradients', toolset='tf-keras-vis'))  # todo: warning + use correct toolset
    # todo: exceptions

