import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import random as rd
import pandas as pd
import plotly.express as px
import googlefinance as finance
from yahooquery import Ticker

pd.set_option("display.max_columns", None)

# Companies listed in the Financial Times Stock Exchange 100 and in the Financial Times Stock Exchange 250

# Read the CSV File
FTSE100 = pd.read_csv("EPIC.csv")

# Share Codes
symbols = FTSE100["Symbol"]
all_symbols = symbols.tolist()
all_symbols = " ".join(all_symbols)


#Reference This
# https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance

print(all_symbols)


# The loop below is extremely inneficient, other solutions may be more viable!!!!!
#for a in all_simbols:
 #    data_spef = yf.Ticker(a)
  #   print(data_spef.info)


# There is a way that is more efficient /////// https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance
# The code below may not be entirely accurate. See website provided above

# Download all info
my_info = yf.download(all_symbols, start="2022-06-01", end="2023-01-01", group_by="ticker")


print(my_info.info)
