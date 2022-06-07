from captum.attr import IntegratedGradients

from methods.Method import Method


class CaptumIntegratedGradients(Method):

    @staticmethod
    def get_method_key():
        return 'integrated_gradients'

    @staticmethod
    def execute(model, *args, **kwargs):
        required_keys = ['input']
        for key in required_keys:
            if key not in kwargs:
                raise Exception(f'Call to {CaptumIntegratedGradients.get_method_key()} requires the argument {key}')
        ig = IntegratedGradients(model)
        if 'target' in kwargs:
            attribution = ig.attribute(kwargs['input'], target=kwargs['target'])
        else:
            attribution = ig.attribute(kwargs['input'])

        return attribution
