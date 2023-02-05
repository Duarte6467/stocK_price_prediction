import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import yfinance as yf
import sklearn as sk
import matplotlib
from matplotlib import pyplot as plt
import pandas_datareader as pdr
import datetime as dt

ticker = 'AAPL'
start = dt.datetime(2019, 1, 1)
end = dt.datetime(2020, 12, 31)

data = pdr.get_data_yahoo('AAPL', start, end).read()

print(data.head())

import pandas_datareader as pdr
import datetime as dt

ticker = ["AAPL", "IBM", "TSLA"]
start = dt.datetime(2019, 1, 1)
end = dt.datetime(2020, 12, 31)
data = pdr.get_data_yahoo(ticker, start, end)
print(data)

pdr.get_data_fred('GS10')

import pandas_datareader.data as web
import pandas as pd
import datetime as dt

df = web.DataReader('GE', 'yahoo', start='2019-09-10', end='2019-10-09')
df.head()

start = dt.datetime(2010, 1, 29)

end = dt.datetime.today()

actions = web.DataReader('GOOG', 'yahoo-actions', start, end)

actions.head()

dividends = web.DataReader('IBM', 'yahoo-dividends', start, end)

dividends.head()

import yfinance as yf

stock = yf.Ticker("AAPL")
stock.calendar

a = stock.history(period="6mo")
a['Daily Spread'] = a['Open'] - a['Close']
a
a['Log returns'] = np.log(a['Close'] / a['Close'].shift())
a
a['Log returns'].std()
volatility = a['Log returns'].std() * 252 / 2 ** .5

str_vol = str(round(volatility, 4) * 100)

fig, ax = plt.subplots()
a['Log returns'].hist(ax=ax, bins=50, alpha=0.6, color='b')
ax.set_xlabel('Log return')
ax.set_ylabel('Freq of log return')
ax.set_title('AAPL volatility: ' + str_vol + '%')