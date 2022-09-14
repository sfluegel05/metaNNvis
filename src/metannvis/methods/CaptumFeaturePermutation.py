from captum.attr import FeaturePermutation

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import FEATURE_PERMUTATION


class CaptumFeaturePermutation(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return FEATURE_PERMUTATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        permutation = FeaturePermutation(model, **init_args)
        attribution = permutation.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
