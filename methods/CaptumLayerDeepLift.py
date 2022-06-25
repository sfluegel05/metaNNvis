from captum.attr import LayerDeepLift

from methods.Method import Method
from methods.method_keys import LAYER_DEEP_LIFT


class CaptumLayerDeepLift(Method):

    @staticmethod
    def get_method_key():
        return LAYER_DEEP_LIFT

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        deep_lift = LayerDeepLift(model, **init_args)
        attribution = deep_lift.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
