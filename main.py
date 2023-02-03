import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import random as rd
import warnings
warnings.filterwarnings("ignore")   # Remove deprecated warnings
import pandas as pd
import plotly.express as px
import googlefinance as finance
from yahooquery import Ticker
import openpyxl
import ray
import sklearn as sk

pd.set_option("display.max_columns", None)

# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE = pd.read_csv("EPIC.csv")

#  Symbol Codes listed in the FTSE100 index / TOP30
symbols = FTSE["Symbol"]

# Convert format to List
all_symbols = symbols.tolist()

# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower
data = yf.download(all_symbols, start= "2018-02-05", end= "2023-02-02" , interval="1d", threads= True, group_by= "ticker")

# Check Summary Statistics for each Symbol
#summary_statistics = data.describe()
#print(summary_statistics)

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
    forward_eps = inf.get("forwardEps")
    forward_PE = inf.get("forwardPE")
    sector_info = sector_info.append({"Symbol": ticker, "Sector": sector, "Forward EPS": forward_eps, "Forward PE": forward_PE}, ignore_index= True)

print(sector_info)

sector_info_csv = sector_info.to_csv("sector_info.csv")

# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
data = pd.merge(data,sector_info, how="inner")
print(data)

# This section, we need to do a groupby.mean or difference, to get the data all neat.
# We can do a groupby("Symbol"), which will average the values for each company, or groupby("Sector"), which will average
# the values of each sector, or we can do it both.







# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic

# Testing / Can remove Later
appl = yf.Ticker("AHT.L")
print(appl.info)