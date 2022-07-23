import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from torchvision import datasets

from torch import optim
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from Main import perform_feature_visualization
from methods import method_keys
from toolsets import toolset_keys


def train_torch_net():
    torch_net = TorchConvNet()
    optimizer = optim.Adam(torch_net.parameters())

    train_data = datasets.CIFAR10(root=os.path.join('..', 'data', 'cifar10_train'), train=True, download=True,
                                  transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = datasets.CIFAR10(root=os.path.join('..', 'data', 'cifar10_test'), train=False, download=True,
                                 transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    for epoch in range(12):
        torch_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = torch_net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 40 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_data),
                           100. * batch_idx / len(train_loader), loss.item()))

        torch_net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = torch_net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_data)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data),
            100. * correct / len(test_data)))

    torch.save(torch_net.state_dict(), os.path.join('..', 'models', 'torch_cifar10.pth'))


def activation_maximization():
    torch_net = TorchConvNet()
    torch_net.load_state_dict(torch.load(os.path.join('..', 'models', 'torch_cifar10.pth')))
    print(torch_net)
    test_data = datasets.CIFAR10(root=os.path.join('..', 'data', 'cifar10_test'), train=False, download=True,
                                 transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    torch_x, torch_y = next(iter(test_loader))

    attr = perform_feature_visualization(torch_net, method_keys.ACTIVATION_MAXIMIZATION, plot=True,
                                         toolset=toolset_keys.TF_KERAS_VIS,
                                         dummy_input=torch_x,
                                         init_args={'model_modifier': ReplaceToLinear()},
                                         exec_args={'score': CategoricalScore(torch_y.tolist()),
                                                    'seed_input': torch.rand(torch_x.size())})


class TorchConvNet(torch.nn.Module):
    def __init__(self):
        super(TorchConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(24, 32, kernel_size=(3, 3))
        self.conv2_drop = torch.nn.Dropout2d()
        self.conv3_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(32 * 6 * 6, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    activation_maximization()
