from captum.attr import Saliency

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import SALIENCY


class CaptumSaliency(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return SALIENCY

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        saliency = Saliency(model)
        attribution = saliency.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
