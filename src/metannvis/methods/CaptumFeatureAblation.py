from captum.attr import FeatureAblation

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import FEATURE_ABLATION


class CaptumFeatureAblation(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return FEATURE_ABLATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        ablation = FeatureAblation(model)
        attribution = ablation.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
