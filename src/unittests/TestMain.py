import unittest

import tensorflow as tf
import os

from tf_keras_vis.utils.scores import CategoricalScore
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.metannvis.toolsets import toolset_keys
from src.metannvis.methods import method_keys

from src.metannvis.Main import execute, perform_attribution


class TestMain(unittest.TestCase):

    def setUp(self):
        self.tf_model = tf.keras.models.load_model(os.path.join('../..', 'models', 'tf_basic_cnn_mnist'))
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
            execute(self.tf_model, method_keys.GRAD_CAM, toolset=toolset_keys.CAPTUM,
                    exec_args={'inputs': test_input_tensor, 'target': test_labels[0].item()})

    def test_missing_exec_args(self):
        test_input_tensor, test_labels = next(iter(self.mnist_test_dataloader))
        with self.assertRaises(Exception):
            # inputs is missing
            execute(self.tf_model, method_keys.INTEGRATED_GRADIENTS, init_args={'multiply_by_inputs': False},
                    exec_args={'target': test_labels[0].item()})

    def test_method_types(self):
        self.mnist_x, self.mnist_y = next(iter(self.mnist_test_dataloader))
        # perform attribution with a feature-vis method (should fail because no attribution method with that key exists)
        with self.assertRaises(Exception):
            perform_attribution('placeholder model', method_keys.ACTIVATION_MAXIMIZATION, toolset_keys.TF_KERAS_VIS,
                                dummy_input=self.mnist_x, plot=True,
                                exec_args={'score': CategoricalScore(self.mnist_y.numpy().tolist()),
                                           'seed_input': self.mnist_x})
