from captum.attr import Saliency

from methods.Method import Method

class CaptumSaliency(Method):

    @staticmethod
    def get_method_key():
        return 'saliency'

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