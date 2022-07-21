import unittest

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from methods import method_keys

from Main import execute, finish_execution_with_layer


class TestIntegratedGradients(unittest.TestCase):
    def setUp(self):
        self.tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))
        self.mnist_test_data = datasets.FashionMNIST(
            root="datasets",
            train=False,
            download=True,
            transform=ToTensor()
        )
        self.mnist_test_dataloader = DataLoader(self.mnist_test_data, batch_size=64, shuffle=True)

        self.labels_map = {
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

    # integrated gradients with regard to input
    def test_primary_integrated_gradients(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        test_input_tensor.requires_grad_()

        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

        # Rescale the images from [0,255] to the [0.0,1.0] range.
        x_train = x_train[..., np.newaxis] / 255.0

        attr = execute(self.tf_model, method_keys.INTEGRATED_GRADIENTS, plot=True,
                       exec_args={'inputs': x_train[:8], 'target': y_train[:8]})


    # layer integrated gradients: verify output for second conv layer (layer only provided after intermediate step)
    def test_layer_integrated_gradients_intermediate(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        test_input_tensor.requires_grad_()
        inter = execute(self.tf_model, method_keys.LAYER_INTEGRATED_GRADIENTS,
                        init_args={'multiply_by_inputs': False},
                        exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})
        res = finish_execution_with_layer(inter, 'Conv_1')

        self.assertTrue(isinstance(res, torch.Tensor))
        self.assertEquals(res.size()[1], 20)
        self.assertEquals(res.size()[2], 8)
        self.assertEquals(res.size()[3], 8)

    # layer integrated gradients: verify output for second conv layer (layer provided directly)
    def test_layer_integrated_gradients_direct(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        test_input_tensor.requires_grad_()
        res = execute(self.tf_model, method_keys.LAYER_INTEGRATED_GRADIENTS, plot=True,
                      init_args={'multiply_by_inputs': False, 'layer': 'Conv_1'},
                      exec_args={'inputs': test_input_tensor[0].unsqueeze(0), 'target': test_labels[0].item()})

        self.assertTrue(isinstance(res, torch.Tensor))
        self.assertEquals(res.size()[1], 20)
        self.assertEquals(res.size()[2], 8)
        self.assertEquals(res.size()[3], 8)

    # nonexistent layer provided -> exception
    def test_layer_integrated_gradients_wrong_layer(self):
        with self.assertRaises(Exception):
            test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
            test_input_tensor.requires_grad_()
            execute(self.tf_model, method_keys.LAYER_INTEGRATED_GRADIENTS,
                    init_args={'multiply_by_inputs': False, 'layer': 'not a layer'},
                    exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})

    # neuron integrated gradients: verify output for neuron in second conv layer
    def test_neuron_integrated_gradients(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        test_input_tensor.requires_grad_()
        res = execute(self.tf_model, method_keys.NEURON_INTEGRATED_GRADIENTS,
                      init_args={'multiply_by_inputs': False, 'layer': 'Conv_1'},
                      exec_args={'inputs': test_input_tensor, 'neuron_selector': (3, 3, 3)})

        self.assertTrue(isinstance(res, torch.Tensor))
        self.assertEquals(res.size()[1], 1)
        self.assertEquals(res.size()[2], 28)
        self.assertEquals(res.size()[3], 28)
