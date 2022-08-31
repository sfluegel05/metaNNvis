from captum.attr import NeuronDeconvolution

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import NEURON_DECONVOLUTION


class CaptumNeuronDeconvolution(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_DECONVOLUTION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        deconv = NeuronDeconvolution(model, **init_args)
        attribution = deconv.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
