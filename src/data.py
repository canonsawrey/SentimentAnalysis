import requests
import requests_cache
import pandas
import datetime

# For reducing API usage
requests_cache.install_cache('github_cache', backend='sqlite', expire_after=180)

# API key used for AlphaVantage requests - should store remotely at some point
av_api_key = "AAGKF1ZHDERDR610"


# Utility function for constructing URL to query AV API
def build_url(security: str) -> str:
    return "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + security + \
           "&outputsize=full&apikey=" + av_api_key


# Returns time series data for the given security
def time_series_data(security: str):
    r = requests.get(url=build_url(security))
    raw = r.json()
    time_series = raw['Time Series (Daily)']
    return time_series


# Returns a df of daily prices for the given security. If no time specified, closing prices used by default
def daily_df(security, time='4. close'):
    time_series = time_series_data(security)
    time_series_daily = {}
    for key in time_series:
        time_series_daily[datetime.datetime.strptime(key, '%Y-%m-%d')] = float(time_series[key][time])
    df = pandas.DataFrame(list(time_series_daily.items()), columns=['Date', 'Price'])
    df.set_index('Date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df

