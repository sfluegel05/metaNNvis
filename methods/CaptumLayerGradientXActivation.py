from captum.attr import LayerGradientXActivation

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import LAYER_GRADIENT_X_ACTIVATION


class CaptumLayerGradientXActivation(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return LAYER_GRADIENT_X_ACTIVATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        grad_x_act = LayerGradientXActivation(model, **init_args)
        attribution = grad_x_act.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
