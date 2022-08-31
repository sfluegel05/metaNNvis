from tf_keras_vis.gradcam import GradcamPlusPlus

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import GRAD_CAM_PLUS_PLUS


class TfKerasVisGradcamPlusPlus(AbstractAttributionMethod):
    @staticmethod
    def get_method_key():
        return GRAD_CAM_PLUS_PLUS

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        gradcam = GradcamPlusPlus(model, **init_args)
        attr = gradcam(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
