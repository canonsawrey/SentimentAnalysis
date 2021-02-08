
"""

Provides an example of how to use a live data source.
Note: this will not work after clone due to missing credentials.

"""

from live_data_source_reddit import LiveDataSourceReddit


data_source = LiveDataSourceReddit(ticker="AMZN")

data_source.load_size(5)

print(data_source.list_data())
