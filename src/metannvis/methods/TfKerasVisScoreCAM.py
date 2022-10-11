from tf_keras_vis.scorecam import Scorecam

from src.metannvis.methods.AbstractAttributionMethod import AbstractAttributionMethod
from src.metannvis.methods.method_keys import SCORE_CAM


class TfKerasVisScoreCAM(AbstractAttributionMethod):
    @staticmethod
    def get_method_key():
        return SCORE_CAM

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        scorecam = Scorecam(model, **init_args)
        attr = scorecam(**exec_args)

        return attr

    @staticmethod
    def get_required_exec_keys():
        return ['score', 'seed_input']
