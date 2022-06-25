from captum.attr import LayerIntegratedGradients

from methods.Method import Method
from methods.method_keys import LAYER_INTEGRATED_GRADIENTS


class CaptumLayerIntegratedGradients(Method):

    @staticmethod
    def get_method_key():
        return LAYER_INTEGRATED_GRADIENTS

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        ig = LayerIntegratedGradients(model, **init_args)
        attribution = ig.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
