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

# Display all collumns
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



# There is a way that is more efficient /////// https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance
# The code below may not be entirely accurate. See website provided above

#--------------------------------------------------------------------------------------------------------
#   This section was made by using the yfinance module ( I believe it is not necessary)
#   Download all info


#my_info = yf.download(all_symbols, start="2022-06-01", end="2023-01-01", group_by="ticker")


# print(my_info.info)

#--------------------------------------------------------------------------------------------------------


print("Main Method",100 * "-")

# Select all symbols of the top 30 companies listed in FTSE100
top30 = Ticker(all_symbols)


# Retrieve all financial data from the top 30 companies listed in the FTSE100
financial_data = top30.all_financial_data()

print(financial_data)


# Save all the information to an Excel format
dataset = financial_data.to_excel("output.xlsx")


