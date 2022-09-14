from src.metannvis.frameworks.TensorFlow2Framework import TensorFlow2Framework
from src.metannvis.methods.TfKerasVisGradcam import TfKerasVisGradcam
from src.metannvis.methods.TfKerasVisLayerCAM import TfKerasVisLayerCAM
from src.metannvis.methods.TfKerasVisSaliency import TfKerasVisSaliency
from src.metannvis.methods.TfKerasVisActivationMaximization import TfKerasVisActivationMaximization
from src.metannvis.methods.TfKerasVisScoreCAM import TfKerasVisScoreCAM
from src.metannvis.methods.TfKervasVisGradcamPlusPlus import TfKerasVisGradcamPlusPlus
from src.metannvis.toolsets.Toolset import Toolset
from src.metannvis.toolsets.toolset_keys import TF_KERAS_VIS


class TfKerasVis(Toolset):

    @staticmethod
    def get_toolset_key():
        return TF_KERAS_VIS

    @staticmethod
    def get_framework():
        return TensorFlow2Framework.get_framework_key()

    @staticmethod
    def get_methods(method_type):
        return filter(lambda x: x.get_method_type() == method_type,
                      [TfKerasVisSaliency, TfKerasVisGradcam, TfKerasVisActivationMaximization,
                       TfKerasVisLayerCAM, TfKerasVisScoreCAM, TfKerasVisGradcamPlusPlus])
