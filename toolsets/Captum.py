from frameworks.PyTorchFramework import PyTorchFramework
from methods.CaptumIntegratedGradients import CaptumIntegratedGradients
from toolsets.Toolset import Toolset


class Captum(Toolset):

    @staticmethod
    def get_toolset_key():
        return 'captum'

    @staticmethod
    def get_framework():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def get_methods():
        return [CaptumIntegratedGradients]