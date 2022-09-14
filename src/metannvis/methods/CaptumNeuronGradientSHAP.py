from captum.attr import NeuronGradientShap

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import NEURON_GRADIENT_SHAP


class CaptumNeuronGradientSHAP(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return NEURON_GRADIENT_SHAP

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        grad_shap = NeuronGradientShap(model, **init_args)
        attribution = grad_shap.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'neuron_selector', 'baselines']
