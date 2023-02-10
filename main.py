import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import random as rd
import warnings
warnings.filterwarnings("ignore")   # Remove deprecated warnings / from pandas for instance
import pandas as pd
import plotly.express as px
from yahooquery import Ticker
import openpyxl
import pandasdmx as pdmx
import sklearn as sk


pd.set_option("display.max_columns", None)

# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE = pd.read_csv("EPIC.csv")

#  Symbol Codes listed in the FTSE100 index / TOP30
symbols = FTSE["Symbol"]

# Convert format to List
all_symbols = symbols.tolist()
print(all_symbols)

# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower
data = yf.download(all_symbols, period="2y" , interval="1d", threads= True, group_by= "ticker")

# Reference this
# https://stackoverflow.com/questions/69117454/from-yfinance-to-manipulate-dataframe-with-pandas
data = data.stack(0).reset_index().rename(columns= {"level_1":"Symbol"})

print(data)


# Experimental Feature using Ray module
#https://stackoverflow.com/questions/73123556/downloading-yahoo-finance-data-much-faster-than-using-just-a-for-loop
#------------------------------------------------------------------------------------------------------------------

 # Get the sector for each symbol
sector_info = pd.DataFrame()

# This part takes ages to run / need to make this efficient
for ticker in all_symbols:
    inf = yf.Ticker(ticker).info
    sector = inf.get("sector")
    sector_info = sector_info.append({"Symbol": ticker, "Sector": sector}, ignore_index= True)

print(sector_info)

sector_info_csv = sector_info.to_csv("sector_info.csv")


# Reference This
#https://aeturrell.github.io/coding-for-economists/data-extraction.html

AAPL = yf.Ticker("aapl")
print(AAPL.info)

# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
final_dataset = pd.merge(data,sector_info, how="inner")
print(final_dataset)

# This section, we need to do a groupby.mean or difference, to get the data all neat.

# Grouped by Sector and Date
grouped_by = final_dataset.groupby(["Sector","Date"]).mean()
print(grouped_by)


# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic
plt.figure()
plt.plot(grouped_by["Date"], grouped_by["Adj Close"])
plt.grid()
plt.show()


# Moving Average
data.loc['mean'] = data.mean()  # insert average row at the bottom pandas

Close = data['Adj Close']
print(Close)
Close = Close.iloc[-1]
print(Close)


# Momentum
MomentumClose = data["Adj Close"]


print(MomentumClose)






# RSI (on n period) = 100 * average of n days up / (average of n days up + average of n days down)

#def RSI():
 #   '''RSI (on n period) = 100 * average of n days up /
  #  (average of n days up + average of n days down)'''

   # Close = data.xs('Adj Close', axis=1, level=1)
    #Close.
    #RSI = pd.DataFrame(columns=Close.columns)
    #T = Close.columns
    #len(T)
    #for i in len(T):
     #   Close.
