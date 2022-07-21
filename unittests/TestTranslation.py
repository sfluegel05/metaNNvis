import logging
import os.path
import unittest

import keras.utils.vis_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tensorflow as tf
import tensorflow_hub as hub

from Main import translate_model
from torch.utils.data import DataLoader
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from frameworks.PyTorchFramework import PyTorchFramework


class TestTranslation(unittest.TestCase):

    def setUp(self):
        self.torch_net = NoDropoutNet()
        self.torch_net.load_state_dict(torch.load('../project_preparation_demo/models/mnist_pytorch_24_06_22_no_dropout.pth'))
        self.alexNet = torchvision.models.alexnet(pretrained=True)

        self.mnist_train = torchvision.datasets.MNIST(os.path.join('..', 'datasets'), download=True,
                                                      transform=torchvision.transforms.ToTensor())
        self.mnist_torch = DataLoader(self.mnist_train, batch_size=64)

    def test_torch_to_tf(self):
        """A torch to tf translation should result (1) in a tf model (2) which has a high accuracy on the same data
        the torch model has been trained on"""
        dummy = torch.randn([10, 3, 224, 224])
        # alexnet_tf = translate(self.alexNet, TensorFlow2Framework.get_framework_key(),
        #                       dummy_input=dummy)
        # self.assertTrue(TensorFlow2Framework.is_framework_model(alexnet_tf))
        mnist_x, mnist_y = next(iter(self.mnist_torch))
        mynet_tf = translate_model(self.torch_net, TensorFlow2Framework.get_framework_key(), dummy_input=mnist_x)
        keras.utils.vis_utils.plot_model(mynet_tf, 'test_translation_2tf.png', show_shapes=True)

        tf_logits = mynet_tf(mnist_x.numpy().reshape((mnist_x.size(dim=0), mnist_x.size(dim=2), mnist_x.size(dim=3),
                                                      mnist_x.size(dim=1))))
        tf_output = np.argmax(tf_logits, axis=1)
        torch_logits = self.torch_net(mnist_x)
        torch_output = torch.argmax(torch_logits, dim=1)
        for torch_l, tf_l in zip(np.array(torch_logits.detach().numpy()).flatten(),
                                 np.array(tf_logits.numpy()).flatten()):
            self.assertAlmostEqual(torch_l, tf_l, 3)

        # for torch_l, torch_o, tf_l, tf_o, y in zip(torch_logits, torch_output, tf_logits, tf_output,
        #                                                          mnist_y.tolist()):
        #    print(f'torch-output: {torch_o}, tf-output: {tf_o}, label: {y}')
        #    print(f'torch-logits: {np.array(torch_l.detach().numpy())}')
        #    print(f'tf-logits: {tf_l}')

    def test_torch_to_tf_feed_forward_net(self):
        data, labels = load_titanic_data()
        data_torch = torch.from_numpy(data).type(torch.FloatTensor)
        labels_torch = torch.from_numpy(labels)
        ffn = TitanicSimpleNNModel()
        ffn.load_state_dict(torch.load(os.path.join('..', 'models', 'titanic_model.pt')))
        ffn_tf = translate_model(ffn, TensorFlow2Framework.get_framework_key(), dummy_input=data_torch)
        print(ffn_tf.summary())
        print(labels.tolist())
        ffn_output = ffn(data_torch)
        ffn_tf_output = ffn_tf(data)
        print(ffn_output)
        print(ffn_tf_output)
        for torch_l, tf_l in zip(np.array(ffn_output.detach().numpy()).flatten(),
                                 np.array(ffn_tf_output.numpy()).flatten()):
            self.assertAlmostEqual(torch_l, tf_l, 3)

    def test_tf_to_torch_basic_cnn(self):
        tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))
        torch_model = translate_model(tf_model, PyTorchFramework.get_framework_key())

        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0

        mnist_x, mnist_y = next(iter(self.mnist_torch))
        torch_logits = torch_model(torch.from_numpy(x_train).float())
        torch_out = torch.argmax(torch_logits, dim=1)
        tf_logits = tf_model(mnist_x.numpy().reshape(64, 28, 28, 1))
        tf_out = np.argmax(tf_logits, axis=1)
        for torch_l, torch_o, tf_l, tf_o, y in zip(torch_logits, torch_out, tf_logits, tf_out, mnist_y.tolist()):
            print(f'torch-output: {torch_o}, tf-output: {tf_o}, label: {y}')
            print(f'torch-logits: {torch_l}')
            print(f'tf-logits: {tf_l}')
            self.assertEqual(tf_o, torch_o)

    def test_tf_to_torch_mobilenet(self):
        model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/5")
        ])
        model.build([None, 128, 128, 3])
        mobilenet_torch = translate_model(model, PyTorchFramework.get_framework_key())
        print(type(mobilenet_torch))
        # ! translation fails, because onnx2torch is not supporting asymmetric padding


    def test_wrong_model(self):
        """If the provided model format does not match one of the frameworks, the translation should fail"""

        with self.assertRaises(Exception):
            translate_model('not a model', TensorFlow2Framework.get_framework_key())


    def test_wrong_framework(self):
        """If the translation output framework does not exist, the translation should fail"""
        with self.assertRaises(Exception):
            translate_model(self.torch_net, 'not a framework', dummy_input=next(iter(self.mnist_torch)))


    def test_tf_to_tf_translation(self):
        tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))
        translation = translate_model(tf_model, TensorFlow2Framework.get_framework_key())

        self.assertEqual(tf_model, translation)


class TorchConvNet(nn.Module):
    def __init__(self):
        super(TorchConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
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


class LayerTorchConvNet(TorchConvNet):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x3 = x2.view(-1, 320)
        x4 = F.relu(self.fc1(x3))
        x5 = F.dropout(x4, training=self.training)
        x6 = self.fc2(x5)
        return [x1, x2, x3, x4, x5, x6, F.log_softmax(x6, dim=1)]

class NoDropoutNet(nn.Module):
    def __init__(self):
        super(NoDropoutNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# model taken from https://captum.ai/tutorials/Titanic_Basic_Interpret
class TitanicSimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12, 8)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lin1_out = self.linear1(x)
        sigmoid_out1 = self.sigmoid1(lin1_out)
        sigmoid_out2 = self.sigmoid2(self.linear2(sigmoid_out1))
        return self.softmax(self.linear3(sigmoid_out2))


def load_titanic_data():
    import pandas as pd
    titanic_data = pd.read_csv(os.path.join('..', 'data', 'titanic3.csv'))
    titanic_data = pd.concat([titanic_data,
                              pd.get_dummies(titanic_data['sex']),
                              pd.get_dummies(titanic_data['embarked'], prefix="embark"),
                              pd.get_dummies(titanic_data['pclass'], prefix="class")], axis=1)
    titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
    titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())
    titanic_data = titanic_data.drop(
        ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest', 'sex', 'embarked', 'pclass'], axis=1)
    # Convert features and labels to numpy arrays.
    labels = titanic_data["survived"].to_numpy()
    titanic_data = titanic_data.drop(['survived'], axis=1)
    feature_names = list(titanic_data.columns)
    data = titanic_data.to_numpy()

    return data, labels
