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

# Display all columns
pd.set_option("display.max_columns", None)



# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE = pd.read_csv("EPIC.csv")

# Share Codes
symbols = FTSE["Symbol"]
all_symbols = symbols.tolist()
all_symbols1 = " ".join(all_symbols)

#Reference This
# https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance


# There is a way that is more efficient /////// https://stackoverflow.com/questions/71161902/get-info-on-multiple-stock-tickers-quickly-using-yfinance
# The code below may not be entirely accurate. See website provided above


print("Main Method",100 * "-")

# Select all symbols of the top 30 companies listed in FTSE100
top30 = Ticker(all_symbols, progress=True)


# Retrieve all financial data from the top 30 companies listed in the FTSE100
financial_data = top30.all_financial_data()

print(financial_data)
print("-"*100)




# Save all the information to an Excel format
dataset = financial_data.to_excel("output.xlsx")




# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower
# Create empty Dataframe
info_we_need = pd.DataFrame()

for symbol in all_symbols:
    all_info = yf.Ticker(symbol).info
    #Add more parameters here, such as low, high , daily , close P/E ratio and EPS, sentiment analysis, etc
    sector = all_info.get("sector")   # Sector Parameter
    info_we_need = info_we_need.append({"Symbol": symbol,"Sector": sector}, ignore_index= True)

print(info_we_need)






