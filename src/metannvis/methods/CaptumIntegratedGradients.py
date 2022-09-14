from captum.attr import IntegratedGradients, LayerIntegratedGradients

from AbstractAttributionMethod import AbstractAttributionMethod
from method_keys import INTEGRATED_GRADIENTS


class CaptumIntegratedGradients(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return INTEGRATED_GRADIENTS

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        if 'layer' in init_args:
            print(model.Conv_0)
            ig = LayerIntegratedGradients(model, layer=model.Conv_0)
        else:
            ig = IntegratedGradients(model, **init_args)
        attribution = ig.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
