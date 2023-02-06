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

# Yfinance module to get all the information that we need / May be the most efficient way to get results, although slower
data = yf.download(all_symbols, period="2y" , interval="1d", threads= True, group_by= "ticker")

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

# Reference This
#https://aeturrell.github.io/coding-for-economists/data-extraction.html

# Gross Domestic Product Data from OECD/ Using its api
# Request Access to OECD API
oecd_data = pdmx.Request("OECD")

GDP = oecd_data.data(
    resource_id = "QNA",
    key = "GBR.B1_GE.GPSA.Q/all?startTime=2018-Q1&endTime=2022-Q4"
).to_pandas()


print(GDP)

# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
final_dataset = pd.merge(data,sector_info, how="inner")
print(final_dataset)

# This section, we need to do a groupby.mean or difference, to get the data all neat.

# Grouped by Sector and Date
grouped_by = final_dataset.groupby(["Sector","Date"]).mean()
print(grouped_by)
# We can do a groupby("Symbol"), which will average the values for each company, or groupby("Sector"), which will average
# the values of each sector, or we can do it both.



# Testing / Can remove Later
appl = yf.Ticker("AHT.L")
print(appl.info)




# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic

plt.figure()

plt.plot(grouped_by["Date"], grouped_by["Adj Close"])
plt.grid()

plt.show()
