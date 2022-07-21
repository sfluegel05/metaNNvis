import os
import unittest
import torch
import torchvision
from tf_keras_vis.utils.scores import Score, CategoricalScore
from torch.utils.data import DataLoader

from Main import perform_attribution
from methods.method_keys import SALIENCY
from toolsets import toolset_keys
from unittests.TestTranslation import NoDropoutNet


class TestSaliency(unittest.TestCase):

    def setUp(self):
        self.mnist_train = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True,
                                                      transform=torchvision.transforms.ToTensor())
        self.mnist_torch = DataLoader(self.mnist_train, batch_size=64)
        self.mnist_x, self.mnist_y = next(iter(self.mnist_torch))

    def test_tf_keras_vis_saliency(self):
        torch_net = NoDropoutNet()
        torch_net.load_state_dict(
            torch.load('../project_preparation_demo/models/mnist_pytorch_24_06_22_no_dropout.pth'))

        res = perform_attribution(torch_net, SALIENCY, toolset_keys.TF_KERAS_VIS, dummy_input=self.mnist_x, plot=True,
                                  exec_args={'score': CategoricalScore(self.mnist_y.numpy().tolist()),
                                             'seed_input': self.mnist_x})

        print(res)
