import pandas as pd
import numpy as np
import xlrd
import os
import matplotlib.pyplot as plt

os.getcwd()

# Load CSV to dataframe
df = pd.read_csv('us_retail_sales.csv', dtype=float) # if any missing values will auto convert to float, so do anyway

# Check first few rows of dataframe
df.head()

# Check basic descriptive stats
df.describe()

# Needed melt to set the axis
df2 = pd.melt(df, id_vars=['YEAR'], value_vars=['JAN', 'FEB','MAR', 'APR',
                                                'MAY', 'JUN', 'JUL', 'AUG',
                                                'SEP', 'OCT', 'NOV', 'DEC'], 
              var_name='Month', value_name='Sales')

# 3 Subplots for every 10 years
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))
for i, ax in enumerate(axs):
    start_year = 1992 + i * 10 # Make sure to set correct start year or will produce blank plots
    end_year = start_year + 9
    data = df2[(df2['YEAR'] >= start_year) & (df2['YEAR'] <= end_year)]
    for year in data['YEAR'].unique():
        year_data = data[data['YEAR'] == year]
        ax.plot(year_data['Month'], year_data['Sales'], label=year)
    ax.set_title('Sales from {} to {}'.format(start_year, end_year))
    ax.legend(loc='upper right', bbox_to_anchor=(-0.15, 1.0)) # Sets label to readable area
plt.show()

import warnings
warnings.filterwarnings('ignore')

# Split into training and test sets
train_data = df2[df2['YEAR'] < 2020]
test_data = df2[(df2['YEAR'] == 2020) & (df2['Month'].isin(['JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])) | 
                (df2['YEAR'] == 2021) & (df2['Month'].isin(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']))]
# This sets the date range, think sql

# Encode month
month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12} # Think of this as adding MonthYr type of bucket
train_data['Month_num'] = train_data['Month'].apply(lambda x: month_dict[x])
test_data['Month_num'] = test_data['Month'].apply(lambda x: month_dict[x])

# Prep data 
X_train = train_data[['YEAR', 'Month_num']].values
y_train = train_data['Sales'].values
X_test = test_data[['YEAR', 'Month_num']].values
y_test = test_data['Sales'].values

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# linear regression model
model1 = LinearRegression()
model1.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Linear Regresstion
y_pred1 = model1.predict(X_test)

# Random Forest
y_pred2 = rf.predict(X_test)

# Linear Regression
rmse1 = mean_squared_error(y_test, y_pred1, squared=False)
print('RMSE:', rmse1)

# Random Forest
rmse2 = mean_squared_error(y_test, y_pred2, squared=False)
print('RMSE:', rmse2)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Needed melt to set the axis
df2 = pd.melt(df, id_vars=['YEAR'], value_vars=['JAN', 'FEB','MAR', 'APR',
                                                'MAY', 'JUN', 'JUL', 'AUG',
                                                'SEP', 'OCT', 'NOV', 'DEC'], 
              var_name='Month', value_name='Sales')

# Encode month 
month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df2['Month_num'] = df2['Month'].apply(lambda x: month_dict[x])

# Split data 
train_data = df2[df2['YEAR'] < 2020]
test_data = df2[(df2['YEAR'] == 2020) & (df2['Month'].isin(['JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])) | 
                (df2['YEAR'] == 2021) & (df2['Month'].isin(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']))]

# Prepare data 
X_train = train_data[['YEAR', 'Month_num', 'Sales']].values
y_train = train_data['Sales'].values
X_test = test_data[['YEAR', 'Month_num', 'Sales']].values
y_test = test_data['Sales'].values

# Reshape 
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build 
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 3)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

#Predict 
y_pred = model.predict(X_test)

#Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
