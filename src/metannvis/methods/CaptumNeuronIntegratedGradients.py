from captum.attr import NeuronIntegratedGradients

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import NEURON_INTEGRATED_GRADIENTS


class CaptumNeuronIntegratedGradients(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_INTEGRATED_GRADIENTS

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        ig = NeuronIntegratedGradients(model, **init_args)
        attribution = ig.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
