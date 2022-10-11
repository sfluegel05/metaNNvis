from captum.attr import LayerActivation

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import LAYER_ACTIVATION


class CaptumLayerActivation(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return LAYER_ACTIVATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        act = LayerActivation(model, **init_args)
        attribution = act.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
