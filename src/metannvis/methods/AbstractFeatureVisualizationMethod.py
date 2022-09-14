from abc import ABC

from AbstractMethod import AbstractMethod


class AbstractFeatureVisualizationMethod(AbstractMethod, ABC):

    @staticmethod
    def get_method_type():
        return 'feature_visualization'
