import os
import unittest
import torch
import torchvision
from tf_keras_vis.utils.scores import CategoricalScore
from torch.utils.data import DataLoader

from Main import perform_attribution, perform_feature_visualization
from methods.method_keys import SALIENCY, GRAD_CAM, ACTIVATION_MAXIMIZATION
from toolsets import toolset_keys
from unittests.TestTranslation import NoDropoutNet


class TestSaliency(unittest.TestCase):

    def setUp(self):
        self.mnist_train = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True,
                                                      transform=torchvision.transforms.ToTensor())
        self.mnist_torch = DataLoader(self.mnist_train, batch_size=8)
        self.mnist_x, self.mnist_y = next(iter(self.mnist_torch))
        self.torch_net = NoDropoutNet()
        self.torch_net.load_state_dict(
            torch.load('../project_preparation_demo/models/mnist_pytorch_24_06_22_no_dropout.pth'))

    def test_tf_keras_vis_saliency(self):
        res = perform_attribution(self.torch_net, SALIENCY, toolset_keys.TF_KERAS_VIS, dummy_input=self.mnist_x,
                                  plot=True, exec_args={'score': CategoricalScore(self.mnist_y.numpy().tolist()),
                                                        'seed_input': self.mnist_x})
        print(res)

    def test_tf_keras_vis_gradcam(self):
        res = perform_attribution(self.torch_net, GRAD_CAM, toolset_keys.TF_KERAS_VIS, dummy_input=self.mnist_x,
                                  plot=True, exec_args={'score': CategoricalScore(self.mnist_y.numpy().tolist()),
                                                        'seed_input': self.mnist_x})
        print(res)

    def test_tf_keras_vis_activation_maximization(self):
        res = perform_feature_visualization(self.torch_net, ACTIVATION_MAXIMIZATION, toolset_keys.TF_KERAS_VIS,
                                            dummy_input=self.mnist_x, plot=True,
                                            exec_args={'score': CategoricalScore(self.mnist_y.numpy().tolist()),
                                                       'seed_input': self.mnist_x})
        print(res)
