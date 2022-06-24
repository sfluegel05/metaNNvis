from captum.attr import InputXGradient

from methods.Method import Method

class CaptumInputXGradient(Method):

    @staticmethod
    def get_method_key():
        return 'input_x_gradient'

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