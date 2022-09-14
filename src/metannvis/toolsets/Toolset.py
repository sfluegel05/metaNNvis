from abc import ABC, abstractmethod


class Toolset(ABC):

    @staticmethod
    @abstractmethod
    def get_toolset_key():
        """
        Returns
        -------
        str
            A key for identifying toolsets (e.g. tf-keras-vis or Captum)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_framework():
        """
        Returns
        -------
        str
            The key of the framework for which the toolset has been developed
        """
        pass

    @staticmethod
    @abstractmethod
    def get_methods(method_type):
        """Filters a list of introspection methods for those matching the method_type.

        Parameters
        ----------
        method_type : {'attribution', 'feature_visualization'}
            The required method type

        Returns
        -------
        list
            A list of method/... classes which belong to this toolset

        """
        pass
