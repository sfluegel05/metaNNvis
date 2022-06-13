from abc import ABC, abstractmethod


class Translation(ABC):

    @staticmethod
    @abstractmethod
    def translate(model, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_input():
        pass

    @staticmethod
    @abstractmethod
    def get_output():
        pass