from captum.attr import GradientShap

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import GRADIENT_SHAP


class CaptumGradientSHAP(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return GRADIENT_SHAP

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        gradient_shap = GradientShap(model, **init_args)
        attribution = gradient_shap.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs', 'baselines']
