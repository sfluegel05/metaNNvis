from abc import ABC

from AbstractMethod import AbstractMethod


class AbstractAttributionMethod(AbstractMethod, ABC):

    @staticmethod
    def get_method_type():
        return 'attribution'
