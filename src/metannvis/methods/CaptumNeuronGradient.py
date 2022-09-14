from captum.attr import NeuronGradient

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import NEURON_GRADIENT


class CaptumNeuronGradient(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_GRADIENT

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        grad = NeuronGradient(model, **init_args)
        attribution = grad.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
