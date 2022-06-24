from abc import ABC, abstractmethod


class Method(ABC):

    @staticmethod
    @abstractmethod
    def get_method_key():
        pass

    @staticmethod
    @abstractmethod
    def execute(model, init_args=None, exec_args=None):
        pass

    @staticmethod
    def get_required_init_keys():
        return []

    @staticmethod
    def get_required_exec_keys():
        return []