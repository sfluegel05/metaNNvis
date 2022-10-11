from captum.attr import InputXGradient

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import INPUT_X_GRADIENT


class CaptumInputXGradient(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return INPUT_X_GRADIENT

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        input_x_grad = InputXGradient(model)
        attribution = input_x_grad.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
