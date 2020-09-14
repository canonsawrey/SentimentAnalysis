"""

Defines the interface for a data source. A data source loads the desired text and sentiment score from somewhere
into memory. The text source can vary, potentially being an API, scraped from a website,
or part of a collection from an online dataset (think Kaggle).

"""

import pandas as pd

# Represents a single entry of data. Sentiment is either 0: bad or 1: good
class Datum:
    def __init__(self, sentence: str, sentiment: int):
        self.sentence = sentence
        self.sentiment = sentiment


# Interface for a source of data to be used to train Models
class DataSource:
    # Prepares and returns DF of data. For example - collect data from an API, load file data into memory, etc.
    def load(self):
        raise NotImplementedError("The method not implemented")

    # Returns a list of Datum
    def list_data(self) -> [Datum]:
        raise NotImplementedError("The method not implemented")

    # Returns a DF of datum
    def df_data(self) -> pd.DataFrame:
        raise NotImplementedError("The method not implemented")



