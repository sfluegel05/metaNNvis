from abc import ABC, abstractmethod


class AbstractMethod(ABC):

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

    # distinguish between attribution and feature visualization methods (and possibly other categories)
    @staticmethod
    @abstractmethod
    def get_method_type():
        pass
