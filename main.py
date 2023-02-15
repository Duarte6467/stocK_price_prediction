import os.path
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import warnings

import seaborn as sns
import pandas as pd
import plotly.express as px
import openpyxl
import sklearn as sk
import os

warnings.filterwarnings("ignore")   # Remove deprecated warnings / from pandas for instance

# Read the CSV File / contains the top 30 companies listed in the FTSE100 Index
FTSE100 = pd.read_csv("EPIC.csv")

#  Symbol Codes listed in the FTSE100 index / TOP30
symbols = FTSE100["Symbol"]

all_symbols = symbols.tolist()  # Convert format to List

# Yfinance module to get all the information that we need / Efficient way to get results, although slower
data = yf.download(all_symbols, period="2y" , interval="1d", threads= True, group_by= "ticker")



# https://stackoverflow.com/questions/69117454/from-yfinance-to-manipulate-dataframe-with-pandas
data = data.stack(0).reset_index().rename(columns= {"level_1":"Symbol"}) # Convert Multiindex to 2D Dataframe

print(data)  # Pre-Visualize Raw Data


 # Get the sector for each symbol
#sector_info = pd.DataFrame()

# This part takes ages to run / need to make this efficient
#for ticker in all_symbols:
#    inf = yf.Ticker(ticker).info
#    sector = inf.get("sector")
#    sector_info = sector_info.append({"Symbol": ticker, "Sector": sector}, ignore_index= True)

#print(sector_info)

#sector_info_csv = sector_info.to_csv("sector_info.csv")

sector_info = pd.read_csv("sector_info.csv")

sector_info = sector_info[["Symbol", "Sector"]]
print(sector_info)

# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
final_dataset = pd.merge(data,sector_info, how="inner")
print(final_dataset)

# This section, we need to do a groupby.mean or difference, to get the data all neat.

# Grouped by Sector and Date
grouped_by = final_dataset.groupby(["Sector","Date"]).mean()
print(grouped_by)


# Plot Graphs to see if there are trends in each sector
print(sector_info.value_counts("Sector"))
# Use the final Dataset
# Make a loop:

final_dataset = final_dataset.groupby("Sector")

sector_names  = final_dataset.groups.keys()




#---------------------------------------------------------------------------------------------------------------------
"""The following code section is just used to create charts and csv files for each sector"""



# Run only once to decrease completion time
for sector in sector_names:
    # Group each sector into one dataset
    each_sector = final_dataset.get_group(sector)

    # Separate each argument for clearer interpretation
    name_of_csv = f"{sector}.csv"
    filepath = os.path.join( name_of_csv)
    each_sector.to_csv(filepath, index = False)




# Fetch a list of each sector included in the directory
files = [f for f in os.listdir() if f.endswith(".csv") and f not in  ["EPIC.csv", "sector_info.csv","output.csv"]]

# Plot each graph with each correspondent CSV file
figsize = (30 , 5 * len(files))

sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows= len(files), ncols= 1, figsize= figsize,  sharex=True)



# Plot adjusted close of every company ( each chart is separated between its sector) against time
for i, file in enumerate(files):
    filepath = os.path.join(file)
    df = pd.read_csv(file)
    for stock_symbol, group in df.groupby("Symbol"):
        sns.lineplot(x="Date", y="Adj Close", data=group, ax=axes[i], label=stock_symbol)

    axes[i].set_title(file, fontsize=16)
    axes[i].set_xlabel("Date", fontsize=10)
    axes[i].set_ylabel("Adj Close", fontsize=10)
    axes[i].set_yscale('log')  # Set the y-axis to logarithmic scale

# Remove hashtag after
#plt.show()

#----------------------------------------------------------------------------------------------------------------------
# Data Manipulation Section
# Sort values by Symbol and date
data = data.sort_values(by=["Symbol", "Date"], ascending= True)
data.set_index("Date", inplace= True)

by_symbol = data.groupby("Symbol")["Adj Close"]



# Technical Indicators that will be used
# Calculate Momentum for each Stock based on Adjusted Close Price
momentum = by_symbol.apply(lambda x: x - x.shift(2))


data = pd.concat([data, momentum.rename("Momentum")], axis = 1)

print(data)

# Moving average added to the dataset
moving_average = data["Adj Close"].rolling(window = 3).mean()









# This is the part where thhe Machine Learning Tecniques ( Logistic Regresion and whatnot) will do its magic



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


