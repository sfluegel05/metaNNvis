import os
import unittest

import seaborn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from Main import perform_attribution, perform_feature_visualization
from methods import method_keys
from toolsets import toolset_keys
from unittests.TestTranslation import NoDropoutNet


class TestCleverHans(unittest.TestCase):

    def setUp(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = get_tf_data()
        self.tf_net = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_clever_hans'))

    def test_captum(self):
        n_samples = 8
        methods = [(method_keys.INTEGRATED_GRADIENTS, 'Integrated Gradients'), (method_keys.SALIENCY, 'Saliency'),
                   (method_keys.DEEP_LIFT, 'Deep Lift'), (method_keys.INPUT_X_GRADIENT, 'Input x Gradient'),
                   (method_keys.FEATURE_ABLATION, 'Feature Ablation'),
                   (method_keys.FEATURE_PERMUTATION, 'Feature Permutation')]  # (method_keys.GRAD_CAM, 'Grad-CAM')]
        figure = plt.figure(figsize=(5 * n_samples, 5 * (len(methods) + 1)))
        counter = 1
        for i in range(n_samples):
            figure.add_subplot(len(methods) + 1, n_samples, counter)
            counter += 1
            plt.title(f'Label: {self.y_test[i]}')
            plt.imshow(self.x_test[i], cmap="gray")
        for m, method_label in methods:
            if m == method_keys.GRAD_CAM:
                attr = perform_attribution(self.tf_net, m, plot=False, toolset=toolset_keys.CAPTUM,
                                           exec_args={'inputs': self.x_test[:n_samples], 'relu_attributions': True,
                                                      'target': self.y_test[:n_samples]})
            else:
                attr = perform_attribution(self.tf_net, m, plot=False, toolset=toolset_keys.CAPTUM,
                                           exec_args={'inputs': self.x_test[:n_samples],
                                                      'target': self.y_test[:n_samples]})
            for i in range(n_samples):
                figure.add_subplot(len(methods) + 1, n_samples, counter)
                counter += 1
                plt.title(f'Captum {method_label}')
                seaborn.heatmap(attr[i].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                                center=0, xticklabels=5, yticklabels=5, square=True,
                                cbar_kws={'shrink': 0.8, 'pad': 0.05})
        plt.savefig('clever_hans_results_captum.png', bbox_inches='tight')
        plt.show()

    def test_captum_gradcam(self):
        n_samples = 8
        counter = 1
        figure = plt.figure(figsize=(5 * n_samples, 10))

        for i in range(n_samples):
            figure.add_subplot(2, n_samples, counter)
            counter += 1
            plt.title(f'Label: {self.y_test[i]}')
            plt.imshow(self.x_test[i], cmap="gray")

        attr = perform_attribution(self.tf_net, method_keys.GRAD_CAM, plot=False, toolset=toolset_keys.CAPTUM,
                                   init_args={'layer': 'Conv_1'},
                                   exec_args={'inputs': self.x_test[:n_samples], 'target': self.y_test[:n_samples],
                                              'relu_attributions': True})
        for i in range(n_samples):
            figure.add_subplot(2, n_samples, counter)
            counter += 1

            plt.title(f'Captum Grad-CAM')
            seaborn.heatmap(attr[i].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                            center=0, xticklabels=5, yticklabels=5, square=True,
                            cbar_kws={'shrink': 0.76, 'pad': 0.05})
        plt.savefig('clever_hans_captum_gradcam.png', bbox_inches='tight')

    plt.show()

    def test_captum_layers(self):
        n_samples = 8
        for m in [method_keys.LAYER_INTEGRATED_GRADIENTS, method_keys.LAYER_DEEP_LIFT,
                  method_keys.LAYER_GRADIENT_X_ACTIVATION, method_keys.LAYER_FEATURE_ABLATION]:
            attr = perform_attribution(self.tf_net, m, plot=True, toolset=toolset_keys.CAPTUM,
                                       init_args={'layer': 'Conv_1'},
                                       exec_args={'inputs': self.x_test[:n_samples], 'target': self.y_test[:n_samples]})

    def test_tf_keras_vis(self):
        torch_net = NoDropoutNet()
        torch_net.load_state_dict(torch.load(os.path.join('..', 'models', 'torch_clever_hans.pth')))

        n_samples = 8
        torch_x = self.x_test[:n_samples]
        torch_x = torch_x.astype(np.single)
        torch_x = torch_x.reshape((torch_x.shape[0], torch_x.shape[3], torch_x.shape[1], torch_x.shape[2]))
        torch_x = torch.from_numpy(torch_x)

        methods = [method_keys.SALIENCY,
                   method_keys.GRAD_CAM,
                   # method_keys.ACTIVATION_MAXIMIZATION
                   ]
        figure = plt.figure(figsize=(5 * n_samples, 5 * (len(methods) + 1)))
        counter = 1
        for i in range(n_samples):
            figure.add_subplot(len(methods) + 1, n_samples, counter)
            counter += 1
            plt.title(self.y_test[i])
            plt.imshow(self.x_test[i], cmap="gray")
        for m in methods:
            if m == method_keys.ACTIVATION_MAXIMIZATION:
                attr = perform_feature_visualization(torch_net, m, plot=False, toolset=toolset_keys.TF_KERAS_VIS,
                                                     dummy_input=torch_x,
                                                     init_args={'model_modifier': ReplaceToLinear()},
                                                     exec_args={'score': CategoricalScore(self.y_test[:8].tolist()),
                                                                'seed_input': torch_x})
            elif m == method_keys.GRAD_CAM:
                attr = perform_attribution(torch_net, m, plot=False, toolset=toolset_keys.TF_KERAS_VIS,
                                           dummy_input=torch_x, init_args={},
                                           exec_args={'score': CategoricalScore(self.y_test[:8].tolist()),
                                                      'seed_input': torch_x, 'normalize_cam': False,
                                                      'expand_cam': False})
            else:
                attr = perform_attribution(torch_net, m, plot=False, toolset=toolset_keys.TF_KERAS_VIS,
                                           dummy_input=torch_x, init_args={'model_modifier': ReplaceToLinear()},
                                           exec_args={'score': CategoricalScore(self.y_test[:8].tolist()),
                                                      'seed_input': torch_x})
            for i in range(n_samples):
                figure.add_subplot(len(methods) + 1, n_samples, counter)
                counter += 1
                if m == method_keys.GRAD_CAM:
                    plt.title(f'tf-keras-vis Grad-CAM')
                elif m == method_keys.SALIENCY:
                    plt.title(f'tf-keras-vis Saliency')
                else:
                    plt.title(f'tf-keras-vis {m}')
                seaborn.heatmap(attr[i].squeeze(), cmap="coolwarm",  # vmin=-attr_total_max, vmax=attr_total_max,
                                center=0, xticklabels=5, yticklabels=5, square=True,
                                cbar_kws={'shrink': 0.76, 'pad': 0.05})
        plt.savefig('clever_hans_results_tfkerasvis.png', bbox_inches='tight')
        plt.show()


def get_tf_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    # Set random labels and a squares in top left corner corresponding to the labels
    rng = np.random.default_rng(724)
    y_train = rng.integers(0, 10, size=y_train.size)
    y_test = rng.integers(0, 10, size=y_test.size)
    for i in range(y_train.size):
        x_train[i, :5, :5] = y_train[i] / 10
    for i in range(y_test.size):
        x_test[i, :5, :5] = y_test[i] / 10

    return (x_train, y_train), (x_test, y_test)
