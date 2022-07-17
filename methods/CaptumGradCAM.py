from captum.attr import LayerGradCam

from methods.AbstractAttributionMethod import AbstractAttributionMethod
from methods.method_keys import GRAD_CAM


class CaptumGradCAM(AbstractAttributionMethod):

    @staticmethod
    def get_method_key():
        return GRAD_CAM

    @staticmethod
    def execute(model, init_args=None, exec_args=None):
        if exec_args is None:
            exec_args = {}
        if init_args is None:
            init_args = {}

        grad_cam = LayerGradCam(model, **init_args)
        attribution = grad_cam.attribute(**exec_args)

        return attribution

    @staticmethod
    def get_required_init_keys():
        return ['layer']

    @staticmethod
    def get_required_exec_keys():
        return ['inputs']
