import os
import unittest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Main import perform_attribution
from methods import method_keys
from toolsets import toolset_keys


class TestCleverHans(unittest.TestCase):

    def setUp(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = get_tf_data()
        self.tf_net = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_clever_hans'))

    def test_integrated_gradients(self):
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
        plt.savefig('clever_hans_results.png')
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
