from tf_keras_vis.layercam import Layercam

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import LAYER_CAM


class TfKerasVisLayerCAM(AbstractAttributionMethod):
    @staticmethod
    def get_method_key():
        return LAYER_CAM

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        cam = Layercam(model, **init_args)
        attr = cam(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
