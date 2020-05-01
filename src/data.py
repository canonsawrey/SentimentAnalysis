import requests
import datetime


av_api_key = "AAGKF1ZHDERDR610"


def build_url(security)-> str:
    return "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + security + \
           "&outputsize=full&apikey=" + av_api_key


def data(time, security):
    r = requests.get(url=build_url(security), params=None)
    raw = r.json()
    time_series = raw['Time Series (Daily)']
    today = datetime.date.today()
    days_, open_, close_, high_, low_ = [], [], [], [], []

    for i in range(0, time):
        day = today - datetime.timedelta(days=i)
        formatted = day.strftime("%Y-%m-%d")
        if formatted in time_series:
            price = time_series[formatted]
            days_.append(day)
            # open_.append(float(price['1. open']))
            # high_.append(float(price['2. high']))
            # low_.append(float(price['3. low']))
            close_.append(float(price['4. close']))

    return days_, close_
