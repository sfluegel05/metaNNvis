from captum.attr import LayerIntegratedGradients

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import LAYER_INTEGRATED_GRADIENTS


class CaptumLayerIntegratedGradients(AbstractAttributionMethod):

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
