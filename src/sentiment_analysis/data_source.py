"""

Defines the interface for a data source. A data source loads the desired text and sentiment score from somewhere
into memory. The text source can vary, potentially being an API, scraped from a website,
or part of a collection from an online dataset (think Kaggle).

"""


class Datum:
    def __init__(self, sentence: str, sentiment: int):
        self.sentence = sentence
        self.sentiment = sentiment


class DataSource:
    # Prepares and returns DF of data. For example - collect data from an API, load file data into memory, etc.
    def load(self):
        raise NotImplementedError("The method not implemented")

    def list_data(self) -> [Datum]:
        raise NotImplementedError("The method not implemented")


