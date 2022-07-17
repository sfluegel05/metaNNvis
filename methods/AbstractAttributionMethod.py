from abc import ABC

from methods.AbstractMethod import AbstractMethod


class AbstractAttributionMethod(AbstractMethod, ABC):

    @staticmethod
    def get_method_type():
        return 'attribution'
