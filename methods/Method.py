from abc import ABC, abstractmethod


class Method(ABC):

    @staticmethod
    @abstractmethod
    def get_method_key():
        pass

    @staticmethod
    @abstractmethod
    def execute(model, *args, **kwargs):
        pass