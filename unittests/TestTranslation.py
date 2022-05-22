import unittest

import tensorflow
import torchvision
import torch
from Main import translate
from torch.utils.data import DataLoader
from frameworks.TensorFlow2Framework import TensorFlow2Framework


class TestTranslation(unittest.TestCase):

    def setUp(self):
        self.vgg16_torch = torchvision.models.vgg16(pretrained=True)
        self.mnist_train = torchvision.datasets.MNIST('datasets', download=True, transform=torchvision.transforms.ToTensor())
        self.mnist_torch = DataLoader(self.mnist_train, batch_size=64)
    def test_torch_to_tf(self):
        """A torch to tf translation should result (1) in a tf model (2) which has a high accuracy on the same data
        the torch model has been trained on"""
        dummy = next(iter(self.mnist_torch))[0]
        vgg16_tf = translate(self.vgg16_torch, TensorFlow2Framework.get_framework_key(),
                             dummy_input=next(iter(self.mnist_torch)))

        self.assertTrue(TensorFlow2Framework.is_framework_model(vgg16_tf))

    def test_wrong_model(self):
        """If the provided model format does not match one of the frameworks, the translation should fail"""
        translation = translate('not a model', TensorFlow2Framework.get_framework_key())

        self.assertFalse(translation)

    def test_wrong_framework(self):
        """If the translation output framework does not exist, the translation should fail"""
        translation = translate(self.vgg16_torch, 'not a framework', dummy_input=next(iter(self.mnist_torch)))

        self.assertFalse(translation)
