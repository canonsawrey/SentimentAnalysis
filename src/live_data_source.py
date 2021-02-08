"""

Defines the interface for a live data source. A live data source loads the desired text from somewhere
into memory. The text source can vary, potentially being an API or scraped from a website

"""

import pandas as pd


# Interface for a source of data to be used to train Models
class LiveDataSource:
    # Prepares and returns DF of data. For example - collect data from an API, load file data into memory, etc.
    def load(self):
        raise NotImplementedError("The method not implemented")

    # Returns a list of Datum
    def list_data(self) -> [str]:
        raise NotImplementedError("The method not implemented")

    # Returns a DF of data
    def df_data(self) -> pd.DataFrame:
        raise NotImplementedError("The method not implemented")
