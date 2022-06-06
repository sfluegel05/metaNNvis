from methods.Method import Method


class CaptumIntegratedGradients(Method):

    @staticmethod
    def get_method_key():
        return 'integrated_gradients'

    @staticmethod
    def execute(model, *args, **kwargs):
        return 'integrated gradients result here'
