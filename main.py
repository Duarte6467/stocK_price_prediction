import os.path
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import warnings
from ta.momentum import RSIIndicator
import seaborn as sns
import pandas as pd
import pandas_ta as ta
import plotly.express as px
import openpyxl
import os
import sklearn as sk
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings("ignore")   # Remove deprecated warnings / from pandas for instance

# Read the CSV File / contains all symbols listed in FTSE100 index
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

# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
final_dataset = pd.merge(data,sector_info, how="inner")

# Grouped by Sector and Date
grouped_by = final_dataset.groupby(["Sector","Date"]).mean()

print(sector_info.value_counts("Sector"))

final_dataset = final_dataset.groupby("Sector")
sector_names  = final_dataset.groups.keys()

#---------------------------------------------------------------------------------------------------------------------
# Plot Graphs to see if there are trends in each sector
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


# Data Manipulation Section
#----------------------------------------------------------------------------------------------------------------------
# Data Manipulation Section
# Sort values by Symbol and date (avoid overlapping of calculations)
data = data.sort_values(by=["Symbol", "Date"], ascending= True)
data.columns = data.columns.astype(str)
data = pd.merge(data, sector_info , how= "inner") #
print(data)

# Technical Indicators that will be used
#Group by Symbol will be used to calculate the following technical indicators
by_symbol = data.groupby("Symbol")["Adj Close"]

# Momentum / Calculate Momentum for each Stock based on Adjusted Close Price with a 2-day lag
momentum = by_symbol.apply(lambda x: x - x.shift(2))
data = pd.concat([data, momentum.rename("Momentum")], axis = 1)

# Moving Average / Calculated by symbol with a 3-day lag / Because it is a multi index DataFrame, we need to reset index
data["Moving Average"] = data.groupby("Symbol")["Adj Close"].rolling(window = 2).mean().reset_index(0,drop = True)

# Volatility / Calculate volatility (2 day lag) by each company
data["Volatility"] = by_symbol.pct_change().rolling(window = 2).std()

# Relative Strength Index (RSI)
RSI_lag = 10 # 10 days lag period
RSI = by_symbol.apply(lambda x : RSIIndicator(x, window= RSI_lag).rsi())
data["RSI"] = RSI.values # Fetch Values
# Information Fetched From : https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#momentum-indicators

# Create a Target Variable, using the calculation of Adj Closing price lag of 1 day, and then convert it to a binary classification
data["Binary Predictor"] = data["Adj Close"].diff().apply(lambda x: 1 if x > 0 else 0)


# Replace NaN values with backward filling method / Best for time-series
data = data.fillna(method= "bfill")


# Now, we need to scale the data , to make it easier to interpret and calculate / We will use log scaling, since it is
# better to read and interpret economic indicators.
# Function that applies the logarithmic scale to specific columns chosen
log_scale = FunctionTransformer(np.log1p , validate= True)

# Apply the logarithmic scaling to the column
columns_to_scale = ["Close","High","Low","Open","Volume"]
data[columns_to_scale] = log_scale.transform(data[columns_to_scale])
print("-"*100)
print("The following section is strictly applied to Machine Learning")

# Define features and label

stock_symbols =  data["Symbol"].unique()



# We needed to make a loop to iterate the logistic regression through each symbol
for symbol in stock_symbols:
    data_symbol = data[data["Symbol"] == symbol]
    X = data_symbol[[ "Close","High","Low", "Open","Volume","Momentum","Moving Average","Volatility","RSI"]]
    x_train , x_test , y_train, y_test = train_test_split(X, data_symbol["Binary Predictor"], test_size= 0.2)

    # Apply the Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(x_train , y_train)

    # Make predictions
    log_reg_prediction = log_reg.predict(x_test)
    log_reg_accuracy = accuracy_score(log_reg_prediction , y_test)

    print("Accuracy for", symbol,":", log_reg_accuracy * 100 , "%")



# Make the Long Short Term Memory Algorithm.







