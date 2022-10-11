from torch.nn import Module

from src.metannvis.frameworks.framework_keys import PYTORCH
from src.metannvis.frameworks.Framework import Framework


class PyTorchFramework(Framework):

    @staticmethod
    def get_framework_key():
        return PYTORCH

    @staticmethod
    def is_framework_model(model):
        return isinstance(model, Module)
