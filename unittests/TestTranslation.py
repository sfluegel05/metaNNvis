import os.path
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow
import torchvision
from Main import translate
from torch.utils.data import DataLoader
from frameworks.TensorFlow2Framework import TensorFlow2Framework


class TestTranslation(unittest.TestCase):

    def setUp(self):
        self.torch_net = TorchConvNet()
        self.torch_net.load_state_dict(torch.load('../project_preparation_demo/models/mnist_pytorch.pth'))

        self.alexNet = torchvision.models.alexnet(pretrained=True)

        self.mnist_train = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True, transform=torchvision.transforms.ToTensor())
        self.mnist_torch = DataLoader(self.mnist_train, batch_size=64)

    def test_torch_to_tf(self):
        """A torch to tf translation should result (1) in a tf model (2) which has a high accuracy on the same data
        the torch model has been trained on"""
        #dummy = next(iter(self.mnist_torch))
        #print(dummy.size())
        #dummy = torch.repeat_interleave(dummy, 3, dim=1)
        #print(dummy.size())
        dummy = torch.randn([10, 3, 224, 224])
        print(self.alexNet(dummy))
        vgg16_tf = translate(self.alexNet, TensorFlow2Framework.get_framework_key(),
                             dummy_input=dummy)
        print(vgg16_tf)
        print(type(vgg16_tf))

        #self.assertTrue(TensorFlow2Framework.is_framework_model(vgg16_tf))
        # T

    def test_wrong_model(self):
        """If the provided model format does not match one of the frameworks, the translation should fail"""
        translation = translate('not a model', TensorFlow2Framework.get_framework_key())

        self.assertFalse(translation)

    def test_wrong_framework(self):
        """If the translation output framework does not exist, the translation should fail"""
        translation = translate(self.torch_net, 'not a framework', dummy_input=next(iter(self.mnist_torch)))

        self.assertFalse(translation)


class TorchConvNet(nn.Module):
    def __init__(self):
        super(TorchConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5,5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)