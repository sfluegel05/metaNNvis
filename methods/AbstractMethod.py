from abc import ABC, abstractmethod


class AbstractMethod(ABC):

    @staticmethod
    @abstractmethod
    def get_method_key():
        """
        Returns
        -------
        str
            A key identifying this method (there might be implementations from different toolsets that share the same
            key)
        """
        pass

    @staticmethod
    @abstractmethod
    def execute(model, init_args=None, exec_args=None):
        """Instantiates and calls an introspection method with the given arguments.

        Parameters
        ----------
        model : any
            The neural network on which to perform the introspection method.
        init_args : dict
            The arguments for the method instantiation.
        exec_args : dict
            The arguments for calling the method.

        Returns
        -------
        array-like
            The introspection method's output

        """
        pass

    @staticmethod
    def get_required_init_keys():
        """
        Returns
        -------
        list
            The keys of initialization arguments required for this method
        """
        return []

    @staticmethod
    def get_required_exec_keys():
        """
        Returns
        -------
        list
            The keys of execution arguments required for this method
        """
        return []

    @staticmethod
    @abstractmethod
    def get_method_type():
        """Distinguish between attribution and feature visualization methods (and possibly other categories).

        Returns
        -------
        {'attribution', 'feature_visualization'}
            The method type. This is used for deciding if a method can be called via 'perform_attribution()' or
            'perform_feature_visualization()'

        """
        pass
