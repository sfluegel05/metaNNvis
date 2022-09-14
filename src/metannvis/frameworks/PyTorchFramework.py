from torch.nn import Module

from framework_keys import PYTORCH
from Framework import Framework


class PyTorchFramework(Framework):

    @staticmethod
    def get_framework_key():
        return PYTORCH

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, Module)
