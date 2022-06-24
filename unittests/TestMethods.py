import unittest

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Main import execute


class TestMethods(unittest.TestCase):

    def test_integrated_gradients(self):
        tf_model = tf.keras.models.load_model(os.path.join('..', 'models', 'tf_basic_cnn_mnist'))
        test_data = datasets.FashionMNIST(
            root="datasets",
            train=False,
            download=True,
            transform=ToTensor()
        )
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        test_input_tensor, test_labels = next(iter(test_dataloader))
        test_input_tensor.requires_grad_()

        labels_map = {
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

        n_rows = 1
        for i in range(n_rows):
            label = test_labels[i].item()
            print(label)
            attr = execute(tf_model, 'integrated_gradients', init_args={'multiply_by_inputs': False},
                           exec_args={'inputs': test_input_tensor, 'target': label})
            attr = attr.detach().numpy()

            img = test_input_tensor[i][0].detach()
            figure = plt.figure(figsize=(20, 20))
            figure.add_subplot(n_rows, 2, i * 2 + 1)
            plt.title(f'Label: {labels_map[label]}')
            plt.axis("off")
            plt.imshow(img, cmap="gray")
            figure.add_subplot(n_rows, 2, i * 2 + 2)
            plt.title(f'Integrated Gradients')
            plt.axis("off")
            plt.imshow(attr[0][0], cmap="gray")
            # plt.savefig(f"integrated_gradients_fashion_mnist_demo_{i}.png")

        plt.show()