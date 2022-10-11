from captum.attr import DeepLift

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import DEEP_LIFT


class CaptumDeepLift(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return DEEP_LIFT

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        deep_lift = DeepLift(model, **init_args)
        attribution = deep_lift.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
