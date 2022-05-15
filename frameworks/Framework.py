from abc import ABC, abstractmethod


class Framework(ABC):

    @staticmethod
    @abstractmethod
    def get_framework_key():
        pass

    @staticmethod
    @abstractmethod
    def is_framework_model(model):
        pass
