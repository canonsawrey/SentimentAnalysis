"""

Defines an implementation for a live data source using reddit.com. You will need to create an app and use the values
in the initialization of the PRAW instance

"""

import pandas as pd
import praw
from live_data_source import LiveDataSource


# Implementation of the live data source interface that relies on the twitter API
class LiveDataSourceReddit(LiveDataSource):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = pd.DataFrame(columns=["text"])
        self.reddit = praw.Reddit(
            client_id="your_id",
            client_secret="your_secret",
            user_agent="your_user_agent"
        )

    # Prepares and returns DF of data. For example - collect data from an API, load file data into memory, etc.
    def load_size(self, entries):
        titles = []
        for result in self.reddit.subreddit("all").search(self.ticker, limit=entries):
            titles.append(result.title)
        self.data = pd.DataFrame(titles, columns=["text"])
        return self.data

    # Prepares and returns DF of data. For example - collect data from an API, load file data into memory, etc.
    def load(self):
        return self.load_size(entries=10)

    # Returns a list of data
    def list_data(self) -> [str]:
        return self.data["text"].tolist()

    # Returns a DF of data
    def df_data(self) -> pd.DataFrame:
        return self.data

