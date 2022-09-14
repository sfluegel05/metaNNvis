import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch

from src.unittests.TestTranslation import NoDropoutNet


def get_tf_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    # Set random labels and a squares in top left corner corresponding to the labels
    rng = np.random.default_rng(724)
    y_train = rng.integers(0, 10, size=y_train.size)
    y_test = rng.integers(0, 10, size=y_test.size)
    for i in range(y_train.size):
        x_train[i, :5, :5] = y_train[i] / 10
    for i in range(y_test.size):
        x_test[i, :5, :5] = y_test[i] / 10

    return (x_train, y_train), (x_test, y_test)


# torch dataset from numpy data
class TorchMNIST(Dataset):
    def __init__(self, x, y):
        x = x.astype(np.single)
        x = x.reshape((x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        self.x = torch.from_numpy(x)
        y = y.astype(np.int64)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_torch_net():
    torch_net = NoDropoutNet()
    optimizer = optim.Adam(torch_net.parameters())

    (x_train, y_train), (x_test, y_test) = get_tf_data()
    train_data = TorchMNIST(x_train, y_train)
    test_data = TorchMNIST(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    for epoch in range(10):
        torch_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = torch_net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 800 == 0:
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

    torch.save(torch_net.state_dict(), os.path.join('../..', 'models', 'torch_clever_hans.pth'))


def train_tf_net():
    (x_train, y_train), (x_test, y_test) = get_tf_data()
    tf_net = get_tf_net()
    tf_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics=['accuracy'])

    history = tf_net.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = tf_net.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')
    tf_net.save(os.path.join('../..', 'models', 'tf_clever_hans'))


def get_tf_net():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(10, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(20, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    train_torch_net()
