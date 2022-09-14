from tf_keras_vis.activation_maximization import ActivationMaximization

from AbstractFeatureVisualizationMethod import AbstractFeatureVisualizationMethod
from method_keys import ACTIVATION_MAXIMIZATION


class TfKerasVisActivationMaximization(AbstractFeatureVisualizationMethod):
    @staticmethod
    def get_method_key():
        return ACTIVATION_MAXIMIZATION

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        act = ActivationMaximization(model, **init_args)
        attr = act(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
