import os
import unittest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from tf_keras_vis.utils.scores import CategoricalScore

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
        methods = [method_keys.INTEGRATED_GRADIENTS, method_keys.SALIENCY, method_keys.DEEP_LIFT,
                   method_keys.INPUT_X_GRADIENT, method_keys.FEATURE_ABLATION, method_keys.FEATURE_PERMUTATION]
        figure = plt.figure(figsize=(5 * n_samples, 5 * (len(methods) + 1)))
        counter = 1
        for i in range(n_samples):
            figure.add_subplot(len(methods) + 1, n_samples, counter)
            counter += 1
            plt.xlabel(self.y_test[i])
            plt.imshow(self.x_test[i], cmap="gray")
        for m in methods:
            attr = perform_attribution(self.tf_net, m, plot=False, toolset=toolset_keys.CAPTUM,
                                       exec_args={'inputs': self.x_test[:n_samples], 'target': self.y_test[:n_samples]})
            print(type(attr))
            for i in range(n_samples):
                figure.add_subplot(len(methods) + 1, n_samples, counter)
                counter += 1
                plt.xlabel(m)
                plt.imshow(attr[i], cmap="gray")
        plt.savefig('clever_hans_results_captum.png')
        plt.show()

    def test_tf_keras_vis(self):
        torch_net = NoDropoutNet()
        torch_net.load_state_dict(torch.load(os.path.join('..', 'models', 'torch_clever_hans.pth')))

        n_samples = 8
        torch_x = self.x_test[:n_samples]
        torch_x = torch_x.astype(np.single)
        torch_x = torch_x.reshape((torch_x.shape[0], torch_x.shape[3], torch_x.shape[1], torch_x.shape[2]))
        torch_x = torch.from_numpy(torch_x)

        methods = [method_keys.SALIENCY, method_keys.GRAD_CAM, method_keys.ACTIVATION_MAXIMIZATION]
        figure = plt.figure(figsize=(5 * n_samples, 5 * (len(methods) + 1)))
        counter = 1
        for i in range(n_samples):
            figure.add_subplot(len(methods) + 1, n_samples, counter)
            counter += 1
            plt.xlabel(self.y_test[i])
            plt.imshow(self.x_test[i], cmap="gray")
        for m in methods:
            if m == method_keys.ACTIVATION_MAXIMIZATION:
                attr = perform_feature_visualization(torch_net, m, plot=False, toolset=toolset_keys.TF_KERAS_VIS,
                                                     dummy_input=torch_x,
                                                     exec_args={'score': CategoricalScore(self.y_test[:8].tolist()),
                                                                'seed_input': torch_x})
            else:
                attr = perform_attribution(torch_net, m, plot=False, toolset=toolset_keys.TF_KERAS_VIS,
                                           dummy_input=torch_x,
                                           exec_args={'score': CategoricalScore(self.y_test[:8].tolist()),
                                                      'seed_input': torch_x})
            print(type(attr))
            for i in range(n_samples):
                figure.add_subplot(len(methods) + 1, n_samples, counter)
                counter += 1
                plt.xlabel(m)
                plt.imshow(attr[i], cmap="gray")
        plt.savefig('clever_hans_results_tfkerasvis.png')
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
