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
# Display all columns

# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE = pd.read_csv("EPIC.csv")

# Share Codes
symbols = FTSE["Symbol"]
all_symbols = symbols.tolist()


# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower

data = yf.download(all_symbols, period="5y", interval="1d", threads= True, group_by= "ticker")


print(data)
print(type(data))



empty = pd.DataFrame()

for ticker in all_symbols:
    ind = yf.Ticker(ticker)
    sector = ind.info["sector"]




print(sector)






# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic

