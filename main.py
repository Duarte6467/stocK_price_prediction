""" This Code was  made in collaboration with Hamze Barreh"""

import os.path
import sklearn.metrics
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
from scipy.stats import uniform
# SKLearn Modules
import sklearn as sk
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score , mean_squared_error ,silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans

# Kera Modules
import keras.backend as K
from keras.callbacks import  EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM , Dense , Dropout
from keras.utils.vis_utils import plot_model
warnings.filterwarnings("ignore")   # Remove deprecated warnings / from pandas for instance

# Read the CSV File / contains all symbols listed in FTSE100 index
FTSE100 = pd.read_csv("EPIC.csv")
#  Symbol Codes listed in the FTSE100 index
symbols = FTSE100["Symbol"]
all_symbols = symbols.tolist()  # Convert format to List
# Yfinance module to get all the information that we need / Efficient way to get results, although slower
data = yf.download(all_symbols, period="2y" , interval="1d", threads= True, group_by= "ticker")

data = data.stack(0).reset_index().rename(columns= {"level_1":"Symbol"}) # Convert Multiindex to 2D Dataframe
# Code Fetched from :  https://stackoverflow.com/questions/69117454/from-yfinance-to-manipulate-dataframe-with-pandas

#---------------------------------------------------------------------------------------------------------------------
# This section is extremely bugged and does not work properly
# Get the sector for each symbol
#sector_info = pd.DataFrame()

# This part takes ages to run / need to make this efficient
#for ticker in all_symbols:
#    inf = yf.Ticker(ticker).info
#    sector = inf.get("sector")
#    sector_info = sector_info.append({"Symbol": ticker, "Sector": sector}, ignore_index= True)

#print(sector_info)
#sector_info_csv = sector_info.to_csv("sector_info.csv")
#----------------------------------------------------------------------------------------------------------------------
# Fetch Sector Information from each Stock Symbol
sector_info = pd.read_csv("sector_info.csv")
sector_info = sector_info[["Symbol", "Sector"]]





#----------------------------------------------------------------------------------------------------------------------
# Optional Code just to plot and interpret some values. Not important for the main body!!
# Merge the Datasets with a common key (variable--- "Symbol") / Can merge more than 2 DataFrames
final_dataset = pd.merge(data,sector_info, how="inner")

# Grouped by Sector and Date
grouped_by = final_dataset.groupby(["Sector","Date"]).mean()

# Give the number of Symbols(companies) in every sector listed in FTSE100
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

#plt.show()




#----------------------------------------------------------------------------------------------------------------------

# Data Manipulation Section
# Sort values by Symbol and date (avoid overlapping of calculations)
data = data.sort_values(by=["Symbol", "Date"], ascending= True)
data.columns = data.columns.astype(str)

# Technical Indicators that will be used / Momentum , Moving Average , Relative Strength Index, Volatility
#Group by Symbol will be used to calculate the following technical indicators
by_symbol = data.groupby("Symbol")["Adj Close"]

# Momentum / Calculate Momentum for each Stock based on Adjusted Close Price with a 2-day lag
momentum = by_symbol.apply(lambda x: x - x.shift(2))
data = pd.concat([data, momentum.rename("Momentum")], axis = 1)

# Moving Average / Calculated by symbol with a 2-day lag / Because it is a multi index DataFrame, we need to reset index
data["Moving Average"] = data.groupby("Symbol")["Adj Close"].rolling(window = 2).mean().reset_index(0,drop = True)

# Volatility / Calculate volatility (2 day lag) by each company
data["Volatility"] = by_symbol.pct_change().rolling(window = 2).std()

# Relative Strength Index (RSI)
RSI_lag = 10  # 10 days lag period
RSI = by_symbol.apply(lambda x : RSIIndicator(x, window= RSI_lag).rsi())
data["RSI"] = RSI.values   # Fetch Values

# Information Fetched From : https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#momentum-indicators



#---------------------------------------------------------------------------------------------------------------------
# Scaling / Normalisation of Data
# Code to scale raw data to logarithmic
log_scale = FunctionTransformer(np.log1p , validate= True)
# Apply the logarithmic scaling to the column
columns_to_scale = ["Close","High","Low","Open","Volume"]
data[columns_to_scale] = log_scale.transform(data[columns_to_scale])
print("-"*100)
print("The following section is strictly applied to Machine Learning")
#--------------------------------------------------------------------------------------------------------

# In this section, the values are averaged by its correspondent Sector and Date, respectively
# Now, we need to average the values of each stock by each sector, giving us an approach of investment by sector
data = pd.merge(data, sector_info , how= "inner")
data = data.groupby(["Sector","Date"]).mean()
data = data.reset_index()  # Reset the index

# Create a Target Variable, using the calculation of Adj Closing price lag of 1 day, and then convert it to a binary classification
data["Binary Predictor"] = data["Adj Close"].diff().apply(lambda x: 1 if x > 0 else 0)

# Replace NaN values with backward filling method / Best for time-series
data = data.fillna(method= "bfill")

# Fetch Unique Sectors
sector_names = data["Sector"].unique()

# Define the parameter distribution to check what is the best C-value
param_dist = {"C" : uniform(0.1 , 20)}
# We needed to make a loop to iterate the logistic regression through each symbol
for sector_0 in sector_names:
    data_symbol = data[data["Sector"] == sector_0]
    X = data_symbol[[ "Close","High","Low", "Open","Volume","Momentum","Moving Average","Volatility","RSI"]]
    x_train , x_test , y_train, y_test = train_test_split(X, data_symbol["Binary Predictor"], test_size= 0.2)

    # Check if the class is balanced or not
    #class_counts = data_symbol["Binary Predictor"].value_counts(normalize= True)*100
    #print(class_counts)

    # Apply the Logistic Regression
    log_reg_c = LogisticRegression()
    # Perform Cross-Validation randomized search
    random_search = RandomizedSearchCV(log_reg_c, param_distributions= param_dist , n_iter=100, cv= 5)
    random_search.fit(x_train, y_train)

    # Make prediction based on the  best C Value found
    log_reg = LogisticRegression(C=random_search.best_params_["C"])
    log_reg.fit(x_train, y_train)
    log_reg_prediction = log_reg.predict(x_test)
    log_reg_accuracy = accuracy_score(log_reg_prediction , y_test)
    log_reg_f1 = f1_score(log_reg_prediction , y_test)

    print("Best C value for ", sector_0, random_search.best_params_["C"])
    print("Accuracy for", sector_0,":", log_reg_accuracy * 100 , "%")
    print("F1 Score for",sector_0,":",log_reg_f1)

#-----------------------------------------------------------------------------------------------------------------------
# Make the Long Short Term Memory Algorithm (LSTM)
# Most of the code and its purpose was retrieved from:
# https://www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/

for sector_1 in sector_names:
    data_symbol_lstm = data[data["Sector"] == sector_1]
    X = data_symbol_lstm[["Close","High","Low","Open","Volume","Momentum","Moving Average","Volatility","RSI"]]

    x_train , x_test , y_train , y_test = train_test_split(X, data_symbol_lstm["Binary Predictor"], test_size=0.2)

    trainX = x_train.to_numpy()
    testX = x_test.to_numpy()

    x_train = trainX.reshape(x_train.shape[0], 1 , x_train.shape[1])
    x_test = testX.reshape(x_test.shape[0], 1 , x_test.shape[1])
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape = (1 ,trainX.shape[1])))
    lstm.add(Dropout(0.18)) # Lower Value than standard to avoid loss in accuracy. May cause overfitting.
    lstm.add(Dense(1, activation="sigmoid"))
    lstm.compile(optimizer="adam",loss = "binary_crossentropy", metrics=["accuracy"]) # Binary Cross entropy is better, since we are using binary classification

    lstm.fit(x_train , y_train , epochs= 50 , batch_size=32)
    y_predict = lstm.predict(x_test)

    # Calculate the Binary CrossEntropy and Accuracy
    loss, accuracy = lstm.evaluate(x_test, y_test)
    f1 = f1_score(y_test, np.round(y_predict))
    print("Binary Cross entropy calculated in sector:", sector_1,":", loss)
    print("Accuracy of Sector", sector_1,":", accuracy *100,"%")
    print("F1 score of Sector", sector_1, ":", f1)

#---------------------------------------------------------------------------------------------------------


# Calculate the K- Nearest Neighbour

# Create empty dictionaries  to store results for each sector
sector_accuracy = {}
sector_best_param = {}
sector_f1_scores = {}

for sector_2 in sector_names:
    data_symbol_k = data[data["Sector"] == sector_2]
    X = data_symbol_k[["Close", "High", "Low", "Open", "Volume", "Momentum", "Moving Average", "Volatility", "RSI"]]
    # Split the data
    x_train , x_test , y_train , y_test = train_test_split(X, data_symbol_k["Binary Predictor"], test_size= 0.2)

    # List of parameters that will be used to improve the Model / Find the best number of neighbours
    parameter_grid = {"n_neighbors": [1,3,5,7,9,11,13]}

    # Start KNN Classifier and the grid search
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn , param_grid= parameter_grid , cv= 5 , scoring="f1_micro")


    # Find the best parameter using each combination of hyperparameters
    grid_search.fit(x_train, y_train)

    # Fetch the best parameter value and the corresponding classifier
    best_parameter = grid_search.best_params_
    knn = grid_search.best_estimator_


    # Initialize prediction process on the testing data
    y_pred = knn.predict(x_test)

    # Calculate performance scores
    accuracy = accuracy_score(y_test , y_pred)
    f1_value = f1_score(y_test , y_pred, average="micro")


    sector_accuracy[sector_2] = accuracy
    sector_best_param[sector_2] = best_parameter
    sector_f1_scores[sector_2] = f1_value


# Print the Results for each sector

for sector_2 in sector_names:
    print("Sector:", sector_2)
    print("Accuracy:", sector_accuracy[sector_2]*100,"%")
    print("Best Parameter:", sector_best_param[sector_2])
    print("F1 Score:" , sector_f1_scores[sector_2])




