from torch.nn import Module

from frameworks.framework_keys import PYTORCH
from frameworks.Framework import Framework


class PyTorchFramework(Framework):

    @staticmethod
    def get_framework_key():
        return PYTORCH

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, Module)
