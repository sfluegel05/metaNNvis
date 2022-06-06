from abc import ABC, abstractmethod


class Toolset(ABC):

    @staticmethod
    @abstractmethod
    def get_toolset_key():
        pass

    @staticmethod
    @abstractmethod
    def get_framework():
        pass

    @staticmethod
    @abstractmethod
    def get_methods():
        pass