from captum.attr import NeuronDeepLift

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import NEURON_DEEP_LIFT


class CaptumNeuronDeepLift(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_DEEP_LIFT

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        deep_lift = NeuronDeepLift(model, **init_args)
        attribution = deep_lift.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
