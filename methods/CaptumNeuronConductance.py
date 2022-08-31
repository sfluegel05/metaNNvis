from captum.attr import NeuronConductance

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import NEURON_CONDUCTANCE


class CaptumNeuronConductance(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_CONDUCTANCE

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        conductance = NeuronConductance(model, **init_args)
        attribution = conductance.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
