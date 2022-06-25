import unittest

import tensorflow as tf
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import toolsets.toolset_keys
from methods import method_keys

from Main import execute


class TestMethods(unittest.TestCase):

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

    def test_nonexistent_method(self):
        with self.assertRaises(Exception):
            execute(self.tf_model, 'not a method')

    # don't throw an exception, but execute method from correct toolset after displaying a warning
    def test_wrong_toolset(self):
        # TODO
        pass

    def test_nonexistent_toolset(self):
        with self.assertRaises(Exception):
            execute(self.tf_model, method_keys.INTEGRATED_GRADIENTS, 'not a toolset')

    def test_multiple_methods_available(self):
        # TODO
        pass

    def test_missing_init_args(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        with self.assertRaises(Exception):
            # gradCAM requires an init argument 'layer'
            execute(self.tf_model, method_keys.GRAD_CAM, toolset=toolsets.toolset_keys.CAPTUM,
                    exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})

    def test_missing_exec_args(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        with self.assertRaises(Exception):
            # inputs is missing
            execute(self.tf_model, method_keys.INTEGRATED_GRADIENTS, init_args={'multiply_by_inputs': False},
                           exec_args={'target': test_labels[0].item()})
