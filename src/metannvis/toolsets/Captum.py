from src.metannvis.frameworks.PyTorchFramework import PyTorchFramework
from src.metannvis.methods.CaptumLayerGradientSHAP import CaptumLayerGradientSHAP
from src.metannvis.methods.CaptumLayerConductance import CaptumLayerConductance
from src.metannvis.methods.CaptumLayerDeepLift import CaptumLayerDeepLift
from src.metannvis.methods.CaptumLayerFeatureAblation import CaptumLayerFeatureAblation
from src.metannvis.methods.CaptumLayerGradientXActivation import CaptumLayerGradientXActivation
from src.metannvis.methods.CaptumLayerIntegratedGradients import CaptumLayerIntegratedGradients
from src.metannvis.methods.CaptumDeepLift import CaptumDeepLift
from src.metannvis.methods.CaptumFeatureAblation import CaptumFeatureAblation
from src.metannvis.methods.CaptumFeaturePermutation import CaptumFeaturePermutation
from src.metannvis.methods.CaptumGradCAM import CaptumGradCAM
from src.metannvis.methods.CaptumIntegratedGradients import CaptumIntegratedGradients
from src.metannvis.methods.CaptumNeuronConductance import CaptumNeuronConductance
from src.metannvis.methods.CaptumNeuronDeconvolution import CaptumNeuronDeconvolution
from src.metannvis.methods.CaptumNeuronDeepLift import CaptumNeuronDeepLift
from src.metannvis.methods.CaptumNeuronFeatureAblation import CaptumNeuronFeatureAblation
from src.metannvis.methods.CaptumNeuronGradient import CaptumNeuronGradient
from src.metannvis.methods.CaptumNeuronGradientSHAP import CaptumNeuronGradientSHAP
from src.metannvis.methods.CaptumNeuronIntegratedGradients import CaptumNeuronIntegratedGradients
from src.metannvis.methods.CaptumDeconvolution import CaptumDeconvolution
from src.metannvis.methods.CaptumSaliency import CaptumSaliency
from src.metannvis.methods.CaptumGradientSHAP import CaptumGradientSHAP
from src.metannvis.methods.CaptumInputXGradient import CaptumInputXGradient
from src.metannvis.methods.CaptumLayerActivation import CaptumLayerActivation
from src.metannvis.toolsets.Toolset import Toolset
from src.metannvis.toolsets.toolset_keys import CAPTUM


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
