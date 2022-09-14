from abc import ABC, abstractmethod


class Framework(ABC):

    @staticmethod
    @abstractmethod
    def get_framework_key():
        """
        Returns
        -------
        str
            a string identifying the framework (e.g. PyTorch, TensorFlow)
        """
        pass

    @staticmethod
    @abstractmethod
    def is_framework_model(model):
        """Determines if a model belongs to this framework.

        Parameters
        ----------
        model : any
            a neural network

        Returns
        -------
        bool
            True if the model belongs to the framework, False if not
        """
        pass
