from captum.attr import IntegratedGradients

from methods.Method import Method


class CaptumIntegratedGradients(Method):

    @staticmethod
    def get_method_key():
        return 'integrated_gradients'

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        required_keys = ['input']
        #for key in required_keys:
        #    if key not in kwargs:
        #        raise Exception(f'Call to {CaptumIntegratedGradients.get_method_key()} requires the argument {key}')
        ig = IntegratedGradients(model, **init_args)
        attribution = ig.attribute(**exec_args)

        return attribution
