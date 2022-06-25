from frameworks.PyTorchFramework import PyTorchFramework
from methods.CaptumLayerIntegratedGradients import CaptumLayerIntegratedGradients
from methods.CaptumDeepLift import CaptumDeepLift
from methods.CaptumFeatureAblation import CaptumFeatureAblation
from methods.CaptumFeaturePermutation import CaptumFeaturePermutation
from methods.CaptumGradCAM import CaptumGradCAM
from methods.CaptumInputXGradient import CaptumInputXGradient
from methods.CaptumIntegratedGradients import CaptumIntegratedGradients
from methods.CaptumSaliency import CaptumSaliency
from toolsets.Toolset import Toolset
from toolsets.toolset_keys import CAPTUM


class Captum(Toolset):

    @staticmethod
    def get_toolset_key():
        return CAPTUM

    @staticmethod
    def get_framework():
        return PyTorchFramework.get_framework_key()

    @staticmethod
    def get_methods():
        return [CaptumIntegratedGradients, CaptumDeepLift, CaptumFeatureAblation, CaptumFeaturePermutation,
                CaptumGradCAM, CaptumInputXGradient, CaptumSaliency, CaptumLayerIntegratedGradients]
