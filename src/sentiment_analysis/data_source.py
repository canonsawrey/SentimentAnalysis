"""

Defines the interface for a data source. A data source loads the desired text and sentiment score from somewhere
into memory. The text source can vary, potentially being an API, scraped from a website,
or part of a collection from an online dataset (think Kaggle).

"""

import pandas as pd
import datetime as dt


class DataSource:
    # Prepares and returns DF od data. For example - collect data from an API, load file data into memory, etc.
    def load(self, start: dt.date = dt.date(1970, 1, 1), end: dt.date = dt.date.today()) -> pd.DataFrame:
        raise NotImplementedError("The method not implemented")


"""

Implementations

"""


class TwitterDataSource(DataSource):
    def load(self, start: dt.date = dt.date(1970, 1, 1), end: dt.date = dt.date.today()) -> pd.DataFrame:
        
