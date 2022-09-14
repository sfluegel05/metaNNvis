from captum.attr import LayerConductance

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import LAYER_CONDUCTANCE


class CaptumLayerConductance(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return LAYER_CONDUCTANCE

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        conductance = LayerConductance(model, **init_args)
        attribution = conductance.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
