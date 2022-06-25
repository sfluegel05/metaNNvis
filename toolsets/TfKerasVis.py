from frameworks.PyTorchFramework import PyTorchFramework
from methods.TfKerasVisSaliency import TfKerasVisSaliency
from toolsets.Toolset import Toolset
from toolsets.toolset_keys import TF_KERAS_VIS


class TfKerasVis(Toolset):

    @staticmethod
    def get_toolset_key():
        return TF_KERAS_VIS

    @staticmethod
    def get_framework():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def get_methods():
        return [TfKerasVisSaliency]