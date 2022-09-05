from frameworks.PyTorchFramework import PyTorchFramework
from methods import CaptumLayerGradientSHAP
from methods.CaptumLayerConductance import CaptumLayerConductance
from methods.CaptumLayerDeepLift import CaptumLayerDeepLift
from methods.CaptumLayerFeatureAblation import CaptumLayerFeatureAblation
from methods.CaptumLayerGradientXActivation import CaptumLayerGradientXActivation
from methods.CaptumLayerIntegratedGradients import CaptumLayerIntegratedGradients
from methods.CaptumDeepLift import CaptumDeepLift
from methods.CaptumFeatureAblation import CaptumFeatureAblation
from methods.CaptumFeaturePermutation import CaptumFeaturePermutation
from methods.CaptumGradCAM import CaptumGradCAM
from methods.CaptumIntegratedGradients import CaptumIntegratedGradients
from methods.CaptumNeuronConductance import CaptumNeuronConductance
from methods.CaptumNeuronDeconvolution import CaptumNeuronDeconvolution
from methods.CaptumNeuronDeepLift import CaptumNeuronDeepLift
from methods.CaptumNeuronFeatureAblation import CaptumNeuronFeatureAblation
from methods.CaptumNeuronGradient import CaptumNeuronGradient
from methods.CaptumNeuronGradientSHAP import CaptumNeuronGradientSHAP
from methods.CaptumNeuronIntegratedGradients import CaptumNeuronIntegratedGradients
from methods.CaptumDeconvolution import CaptumDeconvolution
from methods.CaptumSaliency import CaptumSaliency
from methods.CaptumGradientSHAP import CaptumGradientSHAP
from methods.CaptumInputXGradient import CaptumInputXGradient
from methods.CaptumLayerActivation import CaptumLayerActivation
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
    def get_methods(method_type):
        return filter(lambda x: x.get_method_type() == method_type,
                      [CaptumIntegratedGradients, CaptumLayerIntegratedGradients, CaptumNeuronIntegratedGradients,
                       CaptumSaliency,
                       CaptumDeepLift, CaptumLayerDeepLift, CaptumNeuronDeepLift,
                       CaptumInputXGradient, CaptumLayerGradientXActivation,
                       CaptumFeatureAblation, CaptumLayerFeatureAblation, CaptumNeuronFeatureAblation,
                       CaptumFeaturePermutation,
                       CaptumGradCAM,
                       CaptumDeconvolution, CaptumNeuronDeconvolution,
                       CaptumGradientSHAP, CaptumLayerGradientSHAP, CaptumNeuronGradientSHAP,
                       CaptumLayerActivation,
                       CaptumLayerConductance, CaptumNeuronConductance,
                       CaptumNeuronGradient])
