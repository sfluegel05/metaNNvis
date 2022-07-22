from tf_keras_vis.gradcam import Gradcam

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import GRAD_CAM


class TfKerasVisGradcam(AbstractAttributionMethod):
    @staticmethod
    def get_method_key():
        return GRAD_CAM

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        gradcam = Gradcam(model, **init_args)
        attr = gradcam(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
