av_api_key = "AAGKF1ZHDERDR610"
av_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=" + av_api_key
history_days = 365

import requests
import datetime
import matplotlib
import matplotlib.pyplot as plt

r = requests.get(url = av_url, params = None)
raw = r.json()
time_series = raw['Time Series (Daily)']

today = datetime.date.today()
days_, open_, close_, high_, low_ = [], [], [], [], []

for i in range(0, history_days):
    day = today - datetime.timedelta(days=i)
    formatted = day.strftime("%Y-%m-%d")
    if formatted in time_series:
        price = time_series[formatted]
        days_.append(day)
        open_.append(float(price['1. open']))
        high_.append(float(price['2. high']))
        low_.append(float(price['3. low']))
        close_.append(float(price['4. close']))

#x, y = zip(*sorted(data.items()))

#plot = plt.plot(days_, open_, 'b', label = "Open")
plot = plt.plot(days_, close_, 'g', label = "Close")
#plot = plt.plot(days_, low_, 'r', label = "Low")
#plot = plt.plot(days_, high_, label = "High")

ma_days = 50


plt.legend()

plt.show()
