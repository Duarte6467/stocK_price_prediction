import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import random as rd
import pandas as pd
import plotly.express as px
import googlefinance as finance
from yahooquery import Ticker
import openpyxl
pd.set_option("display.max_columns", None)

# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE = pd.read_csv("EPIC.csv")

#  Symbol Codes listed in the FTSE100 index / TOP30
symbols = FTSE["Symbol"]

# Convert format to List
all_symbols = symbols.tolist()

# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower
data = yf.download(all_symbols, period="5y", interval="1d", threads= True, group_by= "ticker")

# Check Summary Statistics for each Symbol
summary_statistics = data.describe()
print(summary_statistics)

# Reference this
# https://stackoverflow.com/questions/69117454/from-yfinance-to-manipulate-dataframe-with-pandas
data = data.stack(0).reset_index().rename(columns= {"level_1":"Symbol"})

print(data)

 # Get the sector for each symbol
sector_info = pd.DataFrame()

for ticker in all_symbols:
    inf = yf.Ticker(ticker).info
    sector = inf.get("sector")
    sector_info = sector_info.append({"Symbol": ticker, "Sector": sector}, ignore_index= True)

print(sector_info)


# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic

