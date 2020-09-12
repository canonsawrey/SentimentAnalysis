"""

A test file for sentiment module related testing

"""

import data_source

twitter_source = data_source.TwitterDataSource()

print(twitter_source.load())
