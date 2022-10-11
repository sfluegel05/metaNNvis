from tf_keras_vis.saliency import Saliency

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import SALIENCY


class TfKerasVisSaliency(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return SALIENCY

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        saliency = Saliency(model, **init_args)
        attr = saliency(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
