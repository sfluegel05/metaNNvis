import math
import os

import seaborn
import torch
import torchvision
from captum.attr import LayerAttribution
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from torch.utils.data import DataLoader

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import frameworks.framework_keys
import toolsets.toolset_keys
from unittests.TestTranslation import NoDropoutNet
from Main import perform_attribution, translate_model, translate_data
import methods.method_keys as methods


def torch_saliency():
    torch_net = NoDropoutNet()
    torch_net.load_state_dict(
        torch.load('../project_preparation_demo/models/mnist_pytorch_24_06_22_no_dropout.pth'))

    mnist_data = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True,
                                            transform=torchvision.transforms.ToTensor())
    mnist_torch = DataLoader(mnist_data, batch_size=64)
    mnist_x, mnist_y = next(iter(mnist_torch))

    res_captum = perform_attribution(torch_net, methods.SALIENCY, toolset=toolsets.toolset_keys.CAPTUM,
                                     exec_args={'inputs': mnist_x, 'target': mnist_y})
    res_tf_keras_vis = perform_attribution(torch_net, methods.SALIENCY, toolset=toolsets.toolset_keys.TF_KERAS_VIS,
                                           dummy_input=mnist_x, init_args={'model_modifier': ReplaceToLinear()},
                                           exec_args={'score': CategoricalScore(mnist_y.tolist()),
                                                      'seed_input': mnist_x, 'normalize_map': False})
    n_samples = 8  # mnist_x.size()[0]
    plot(mnist_x, mnist_y, res_captum, res_tf_keras_vis, n_samples, 'Saliency', 'comparison_torch_saliency.png')


def torch_gradcam(interpolate=False):
    torch_net = NoDropoutNet()
    torch_net.load_state_dict(
        torch.load('../project_preparation_demo/models/mnist_pytorch_24_06_22_no_dropout.pth'))

    torch_conv2 = torch_net.conv2
    mnist_data = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True,
                                            transform=torchvision.transforms.ToTensor())
    mnist_torch = DataLoader(mnist_data, batch_size=64)
    mnist_x, mnist_y = next(iter(mnist_torch))

    res_captum = perform_attribution(torch_net, methods.GRAD_CAM, toolset=toolsets.toolset_keys.CAPTUM,
                                     init_args={'layer': torch_conv2},
                                     exec_args={'inputs': mnist_x, 'target': mnist_y, 'relu_attributions': True})
    if interpolate:
        res_captum = LayerAttribution.interpolate(res_captum, (28, 28))
    res_tf_keras_vis = perform_attribution(torch_net, methods.GRAD_CAM, toolset=toolsets.toolset_keys.TF_KERAS_VIS,
                                           dummy_input=mnist_x,
                                           exec_args={'score': CategoricalScore(mnist_y.tolist()),
                                                      'expand_cam': interpolate,
                                                      'seed_input': mnist_x, 'normalize_cam': False})
    n_samples = 8  # mnist_x.size()[0]
    file_name = 'comparison_torch_gradcam_scaled_to_original.png' if interpolate else 'comparison_torch_gradcam.png'
    plot(mnist_x, mnist_y.tolist(), res_captum, res_tf_keras_vis, n_samples, 'GradCAM', file_name)


def tf_saliency():
    tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))

    (mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
    mnist_x = mnist_x[..., np.newaxis] / 255.0

    res_captum = perform_attribution(tf_model, methods.SALIENCY, toolset=toolsets.toolset_keys.CAPTUM,
                                     exec_args={'inputs': mnist_x[:64], 'target': mnist_y[:64]})
    res_tf_keras_vis = perform_attribution(tf_model, methods.SALIENCY, toolset=toolsets.toolset_keys.TF_KERAS_VIS,
                                           dummy_input=mnist_x, init_args={},
                                           exec_args={'score': CategoricalScore(mnist_y[:64].tolist()),
                                                      'seed_input': mnist_x[:64], 'normalize_map': False})
    n_samples = 8  # mnist_x.shape[0]
    plot(mnist_x, mnist_y, res_captum, res_tf_keras_vis, n_samples, 'Saliency',
         'comparison_tf_saliency_no_model_modifier.png')


def tf_gradcam(interpolate=False):
    tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))

    (mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
    mnist_x = mnist_x[..., np.newaxis] / 255.0

    torch_model = translate_model(tf_model, frameworks.framework_keys.PYTORCH)
    torch_exec_args = translate_data({'inputs': mnist_x[:64], 'target': mnist_y[:64], 'relu_attributions': True},
                                     frameworks.framework_keys.PYTORCH, tf_model, torch_model)
    # torch_exec_args['inputs'] is equal to mnist_x, output of tf_model and torch_model is equal
    res_captum = perform_attribution(torch_model, methods.GRAD_CAM, toolset=toolsets.toolset_keys.CAPTUM,
                                     init_args={'layer': torch_model.Relu_1},
                                     exec_args=torch_exec_args)
    if interpolate:
        res_captum = LayerAttribution.interpolate(res_captum, (28, 28))
    res_tf_keras_vis = perform_attribution(tf_model, methods.GRAD_CAM, toolset=toolsets.toolset_keys.TF_KERAS_VIS,
                                           dummy_input=mnist_x[:64], init_args={},
                                           exec_args={'score': CategoricalScore(mnist_y[:64].tolist()),
                                                      'expand_cam': interpolate, 'penultimate_layer': 'conv2d_1',
                                                      'seed_input': mnist_x[:64], 'normalize_cam': False})
    n_samples = 8  # mnist_x.shape[0]
    file_name = 'comparison_tf_gradcam_scaled_to_original.png' if interpolate else 'comparison_tf_gradcam.png'
    if not isinstance(res_tf_keras_vis, np.ndarray):
        res_tf_keras_vis = res_tf_keras_vis.numpy()
    plot(mnist_x, mnist_y, res_captum, res_tf_keras_vis, n_samples, 'GradCAM', file_name)


def plot(data_x, data_y, res_captum, res_tf_keras_vis, n_samples, methodname, filename):
    if isinstance(res_captum, torch.Tensor):
        res_captum = res_captum.detach()
    else:
        res_captum = torch.from_numpy(res_captum)
    figure = plt.figure(figsize=(5 * n_samples, 5 * 4))
    counter = 1
    attr_max = max(res_captum[:n_samples].max().abs().item(), abs(res_tf_keras_vis[:n_samples].max()))
    attr_min_abs = max(res_captum[:n_samples].min().abs().item(), abs(res_tf_keras_vis[:n_samples].min()))
    attr_total_max = max(attr_max, attr_min_abs)
    diff_max = (res_captum[:n_samples].squeeze() - res_tf_keras_vis[:n_samples]).max().abs().item()
    diff_min_abs = (res_captum[:n_samples].squeeze() - res_tf_keras_vis[:n_samples]).min().abs().item()
    diff_total_max = max(diff_max, diff_min_abs)
    for i in range(n_samples):
        figure.add_subplot(4, n_samples, counter)
        plt.title(data_y[i])
        plt.imshow(data_x[i].squeeze(), cmap="gray")
        figure.add_subplot(4, n_samples, counter + n_samples)
        plt.title(f'Captum {methodname}')
        seaborn.heatmap(res_captum[i].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                        center=0, xticklabels=5, yticklabels=5)
        figure.add_subplot(4, n_samples, counter + 2 * n_samples)
        plt.title(f'tf-keras-vis {methodname}')
        seaborn.heatmap(res_tf_keras_vis[i], cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                        center=0, xticklabels=5, yticklabels=5)
        figure.add_subplot(4, n_samples, counter + 3 * n_samples)
        plt.title('Captum - tf-keras-vis')
        seaborn.heatmap(res_captum[i].squeeze() - res_tf_keras_vis[i],  # vmin=-diff_total_max, vmax=diff_total_max,
                        cmap="coolwarm", center=0, xticklabels=5, yticklabels=5)
        # figure.add_subplot(6, n_samples, counter + 4 * n_samples)
        # plt.title('Captum / (Captum + tf-keras-vis)')
        # seaborn.heatmap(res_captum[i].squeeze() / (res_captum[i].squeeze() + res_tf_keras_vis[i]), cmap="coolwarm",
        #                center=0.5,
        #                xticklabels=5, yticklabels=5)
        # figure.add_subplot(6, n_samples, counter + 5 * n_samples)
        # plt.title('Captum / tf-keras-vis')
        # seaborn.heatmap(res_captum[i].squeeze() / res_tf_keras_vis[i], cmap="coolwarm", center=1,
        #                xticklabels=5, yticklabels=5)
        counter += 1
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def ceil_power_of_10(x):
    e_counter = 0
    while x > 1:
        e_counter += 1
        x /= 10.
    while x <= 0.1:
        e_counter -= 1
        x *= 10.
    x = math.ceil(x)
    while e_counter > 0:
        e_counter -= 1
        x *= 10.
    while e_counter < 0:
        e_counter += 1
        x /= 10.
    return x


if __name__ == '__main__':
    tf_gradcam()
    tf_gradcam(True)
    torch_gradcam()
    torch_gradcam(True)
    torch_saliency()
    tf_saliency()
