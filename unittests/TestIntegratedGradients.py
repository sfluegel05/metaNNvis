import unittest

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import toolsets.toolset_keys
from methods import method_keys

from Main import execute, finish_execution_with_layer
from translations.Tf2TorchTranslation import Tf2TorchTranslation


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

        n_rows = 1
        for i in range(n_rows):
            label = test_labels[i].item()
            layer = self.tf_model.get_layer('conv2d_1')
            attr = execute(self.tf_model, method_keys.INTEGRATED_GRADIENTS, init_args={'layer': self.tf_model},
                           exec_args={'inputs': test_input_tensor, 'target': label})
            attr = attr.detach().numpy()

            img = test_input_tensor[i][0].detach()
            figure = plt.figure(figsize=(20, 20))
            figure.add_subplot(n_rows, 2, i * 2 + 1)
            plt.title(f'Label: {self.labels_map[label]}')
            plt.axis("off")
            plt.imshow(img, cmap="gray")
            figure.add_subplot(n_rows, 2, i * 2 + 2)
            plt.title(f'Integrated Gradients')
            plt.axis("off")
            plt.imshow(attr[0][0], cmap="gray")
            # plt.savefig(f"integrated_gradients_fashion_mnist_demo_{i}.png")

        plt.show()

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
        res = execute(self.tf_model, method_keys.LAYER_INTEGRATED_GRADIENTS,
                        init_args={'multiply_by_inputs': False, 'layer': 'Conv_1'},
                        exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})

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

