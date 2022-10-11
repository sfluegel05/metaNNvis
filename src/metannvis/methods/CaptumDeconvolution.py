from captum.attr import Deconvolution

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import DECONVOLUTION


class CaptumDeconvolution(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return DECONVOLUTION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        deconv = Deconvolution(model)
        attribution = deconv.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
