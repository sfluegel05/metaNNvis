from torch.nn import Module
from torchvision import models

from frameworks.Framework import Framework


class PyTorchFramework(Framework):

    @staticmethod
    def get_framework_key():
        return 'torch'

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, Module)
