"""

Defines the interface for a "model". A model is constructed using a DataSource and is able to then make predictions
from future input

"""

from training_data_source import TrainingDataSource


# Interface for a NLP Model
class Model:
    # Builds a model_info object from a DataSource
    def build_from_data_source(self, tds: TrainingDataSource):
        raise NotImplementedError("The method not implemented")

    # Classifies the sentence; returns tuple of form (classification, prob(classfication))
    def classify(self, sentence: str):
        raise NotImplementedError("The method not implemented")
