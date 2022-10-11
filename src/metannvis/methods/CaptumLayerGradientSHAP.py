from captum.attr import LayerGradientShap

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import LAYER_GRADIENT_SHAP


class CaptumLayerGradientSHAP(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return LAYER_GRADIENT_SHAP

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        grad_shap = LayerGradientShap(model, **init_args)
        attribution = grad_shap.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'baselines']
