from captum.attr import NeuronFeatureAblation

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import NEURON_FEATURE_ABLATION


class CaptumNeuronFeatureAblation(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_FEATURE_ABLATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        feature_abl = NeuronFeatureAblation(model, **init_args)
        attribution = feature_abl.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector']
