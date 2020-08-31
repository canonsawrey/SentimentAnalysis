"""

Defines the interface for a dataset. A dataset contains text and corresponding scores that can be used
to train a model. A common example would tweets containing the 'AAPL' string, followed by the average return for
Apple stock over the following two weeks.

"""

# Modules
import datetime as dt
import pandas as pd
# Local files
import data_source as ds


class DataSet:

    # The data that is loaded from the data_source, stored a pd DF with columns "data", "text", "score"
    data_: pd.DataFrame

    def __init__(self, data_source_: ds.DataSource):
        self.data_ = data_source_.load()

    # Provides the date at which time the dataset starts
    def data_start(self):
        raise NotImplementedError("The method not implemented")

    # Returns a data frame of data between the start and end date
    def data(self, start: dt.date = dt.date(1970, 1, 1), end: dt.date = dt.date.today()) -> pd.DataFrame:
        is_in_date_range = start < self.data_["data"] < end
        return self.data_[is_in_date_range]
