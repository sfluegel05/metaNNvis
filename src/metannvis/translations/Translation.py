from abc import ABC, abstractmethod


class Translation(ABC):

    @staticmethod
    @abstractmethod
    def translate_model(model, **kwargs):
        """Translates a model from one framework to another.

        Parameters
        ----------
        model : any
            The input neural network
        kwargs : dict
            Additional translation arguments (e.g., 'dummy input' for Torch2Tf)

        Returns
        -------
        translated model
            A model in the target framework which is (nearly) equivalent to the input model

        """
        pass

    @staticmethod
    @abstractmethod
    def translate_data(data, **kwargs):
        """Translates data from one framework to another.

        Parameters
        ----------
        data : any
            The input data (e.g. a PyTorch Tensor for the Torch2Tf translation)
        kwargs : dict
            Additional translation arguments (e.g., 'model' for Tf2Torch)

        Returns
        -------
        translated data
            The input data, but in a format that is compatible with the target framework

        """
        pass

    @staticmethod
    @abstractmethod
    def get_input():
        """
        Returns
        -------
        str
            The key of the framework to which a model that can be translated has to belong
        """
        pass

    @staticmethod
    @abstractmethod
    def get_output():
        """
        Returns
        -------
        str
            The key of the framework into which a model will be translated.
        """
        pass
