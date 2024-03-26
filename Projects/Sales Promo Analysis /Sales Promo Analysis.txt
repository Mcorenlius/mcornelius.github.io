import pandas as pd
import numpy as np
import os
import xlrd
import matplotlib.pyplot as plt 

os.getcwd()

df = pd.read_excel('TermProjectSalesTransactions.xlsx')
df.head()

# More readable column headers
df= df.rename(columns={'TransactionNo':'Transaction Number', 
                       'ProductNo': 'Product Number', 
                       'ProductName': 'Product Name',
                       'Quantity': 'Quantity Sold',
                      'CustomerNo': 'Customer Number'})
df.head()

df['Month'] = df['Date'].dt.month

salesbymonth = df.groupby('Month')['Quantity Sold'].sum()

# plot line chart
plt.plot(salesbymonth.index, salesbymonth.values)

# add title and axis labels
plt.title("Sales by MonthNum")
plt.xlabel("MonthNum")
plt.ylabel("Sales")

# show plot
plt.show()

df1 = df

# How many rows in dataframe
df1.shape[0] 

# See summary stats
df1.describe()

# Outlier? 
df1[df1['Quantity Sold'] == 80995]

# Outlier?
df1[df1['Quantity Sold'] == -80995]

#Prob not an outlier
df1[df1['Price'] == 660.62].head()

import random
# Add Sentiment rating

nrows = len(df1)
nones = int(nrows * 0.9)
nzeros = nrows - nones

sentvalues = np.concatenate((np.ones(nones, dtype=int), np.zeros(nzeros, dtype=int)))
np.random.shuffle(sentvalues)

df1['Sentiment Rating'] = sentvalues
df1.head()

# Seeing if the ratio is correct
df1['Sentiment Rating'].value_counts()

# See Agg list of products by sums
productsum = df1.groupby('Product Name')['Quantity Sold'].sum().reset_index()
productsum = productsum.sort_values('Quantity Sold', ascending=False)
productsum.head()

# See Aggr list of countries by product quantity
countrysum = df1.groupby('Country')['Quantity Sold'].sum().reset_index()
countrysum = countrysum.sort_values('Quantity Sold', ascending=False)
countrysum.head()

df.head()

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# label encoding convert Country column nums
le = LabelEncoder()

# Create Empty DF,thhe add too
encoded_countries_df = pd.DataFrame()
encoded_countries_df['Country'] = df1['Country']
encoded_countries_df['Encoded Country'] = le.fit_transform(df1['Country'])
df1['Country'] = le.fit_transform(df1['Country'])

# One-hot encoding convert Date column to multiple cols to diff time intervals
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Month'] = df1['Date'].dt.month
df1 = pd.get_dummies(df1, columns=['Month'])

# Split data 
train = df1.sample(frac=0.8, random_state=42)
test = df1.drop(train.index)

# Define the features and target variable
features = [col for col in df1.columns if col.startswith('Month')] # referencing dummies
target = 'Quantity Sold'

# Random forest regressor on training data
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train[features], train[target])

# Evaluate model on test data
predictions = rf.predict(test[features])
mse = ((predictions - test[target]) ** 2).mean()
print(f"Mean Squared Error: {mse}")

# Feature importances
importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
print(importance)

import warnings
warnings.filterwarnings('ignore')

# Groupings
groups = df1.groupby(['Country', 'Date'])
predictions = pd.DataFrame({'Country': [], 'Date': [], 'Quantity Sold': []})
for (country, date), group in groups:
    row = {'Country': country, 'Date': date, 'Quantity Sold': rf.predict(group[features].mean().to_frame().T)[0]}
    predictions = predictions.append(row, ignore_index=True)

# Predicted quantity sold for countries and month
predictions.head()

predictions['CountryName'] = np.zeros(len(predictions.index))
predictions.head()

# Dict mapping encoded countries to country names
country_map = dict(zip(encoded_countries_df['Encoded Country'], encoded_countries_df['Country']))

# Map function to replace encoded country codes with country names
predictions['CountryName'] = predictions['Country'].map(country_map)

# Quick View
predictions[predictions['CountryName'] == 'Australia'].sort_values('Date').head()

# Separate dataframe, rows where Country == ? 
subset_df = df1[df1['Country'] == 0][['Date', 'Quantity Sold', 'Country']]

# groupby 
grouped_df = subset_df.groupby(['Country', 'Date'])['Quantity Sold'].sum().reset_index()

# print 
print(grouped_df.sort_values('Quantity Sold', ascending=False).head())

df = pd.read_excel('TermProjectSalesTransactions.xlsx')
df= df.rename(columns={'TransactionNo':'Transaction Number', 
                       'ProductNo': 'Product Number', 
                       'ProductName': 'Product Name',
                       'Quantity': 'Quantity Sold',
                      'CustomerNo': 'Customer Number'})

from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

df = pd.get_dummies(df, columns=['MonthYr'])
df.head()

# split the dataset 
X = df.drop(['Quantity Sold', 'Customer Number', 'Transaction Number', 'Product Number', 'Product Name', 'Date'
            , 'Price', 'Country'], axis=1).values
y = df['Quantity Sold'].values

# Label encoding To nums
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

# standardize feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# loop through each country, train and evaluate a separate MLP model
countries = np.unique(df['Country'])
for country in countries:
    # filter the data for the current country
    X_country = X[df['Country'] == country]
    y_country = y[df['Country'] == country]
    
    # split the data into training and testing sets
    train_size = int(0.7 * len(X_country))
    X_train, X_test = X_country[:train_size], X_country[train_size:]
    y_train, y_test = y_country[:train_size], y_country[train_size:]

# MLP model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# train model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# evaluate model
loss = model.evaluate(X_test, y_test, verbose=0)
print('Country:', country, 'Test Loss:', loss)

# weights of first layer
weights = model.layers[0].get_weights()[0]

# get the feature importance by summing the absolute weights of each feature
feature_importance = abs(weights).sum(axis=1)

# Convert X back to a DataFrame
features =['MonthYr_2018-12', 'MonthYr_2019-01',
       'MonthYr_2019-02', 'MonthYr_2019-03', 'MonthYr_2019-04',
       'MonthYr_2019-05', 'MonthYr_2019-06', 'MonthYr_2019-07',
       'MonthYr_2019-08', 'MonthYr_2019-09', 'MonthYr_2019-10',
       'MonthYr_2019-11', 'MonthYr_2019-12']

for i in range(len(features)):
    print(f"{features[i]}: {feature_importance[i]}")

import random
df = pd.read_excel('TermProjectSalesTransactions.xlsx')
df= df.rename(columns={'TransactionNo':'Transaction Number', 
                       'ProductNo': 'Product Number', 
                       'ProductName': 'Product Name',
                       'Quantity': 'Quantity Sold',
                      'CustomerNo': 'Customer Number'})
# Add Sentiment rating

nrows = len(df1)
nones = int(nrows * 0.9)
nzeros = nrows - nones

sentvalues = np.concatenate((np.ones(nones, dtype=int), np.zeros(nzeros, dtype=int)))
np.random.shuffle(sentvalues)

df['Sentiment Rating'] = sentvalues
df['MonthYr'] = df['Date'].dt.year.astype(str) + '-' + df['Date'].dt.month.astype(str).str.zfill(2)
df.head()

df2 = pd.get_dummies(df, columns=['MonthYr', 'Product Name', 'Country'])
df2.head()

df2[df2.isna().any(axis=1)].head()

df3 = df2.dropna()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split the dataset 
Xtrain, Xtest, ytrain, ytest = train_test_split(df3.drop(['Sentiment Rating', 'Transaction Number',
                                                        'Date', 'Product Number', 'Customer Number',
                                                         'Quantity Sold'], axis=1), 
                                                df3['Sentiment Rating'], test_size=0.2, random_state=42)

# Logistic regression model
LR = LogisticRegression()

# Reshape 
ytrain = ytrain.values.reshape(-1, 1)
ytest = ytest.values.reshape(-1, 1)

# Fit model 
LR.fit(Xtrain, ytrain)

# Predict 
ypred = LR.predict(Xtest)

# Evaluate the model
accuracy = LR.score(Xtest, ytest)
print("Accuracy:", accuracy)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluation metrics
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

# Evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# confusion matrix
cm = confusion_matrix(ytest, ypred)

# Plot
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import numpy as np

# Get coefficients
coef = LR.coef_[0]

# Get the corresponding feature names
features = Xtrain.columns

# Sort the coefficients in descending order
sorted_coef_index = np.argsort(coef)[::-1]

