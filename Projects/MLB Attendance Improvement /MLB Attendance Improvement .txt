import pandas as pd
import numpy as np
import os
import xlrd

df = pd.read_csv('dodgers-2022.csv')
df.head()

# Checking summary stats
df.describe()

# Sorting dataframe to see attendence spread
df.sort_values(by='attend', ascending=False)[0:10]

# Find 75th percentile of attendance column
pct75th = df['attend'].quantile(0.75)

# Making new df where all results are the 75th and above based on attendance
df75gt = df[df['attend'] >= pct75th]

df75gt.sort_values(by='attend', ascending=False)[0:10]

# Find 25th percentile of attendance column
pct25th = df['attend'].quantile(0.25)

# Making new df where all results are the 25th and below based on attendance
df25lt = df[df['attend'] <= pct25th]

df25lt.sort_values(by='attend', ascending=False)[0:10]

import seaborn as sns

# Make correlation matrix
cmatrix = df.corr()

# Make heatmap
sns.heatmap(cmatrix, annot=True)

# Find all categorical columns
catcols = df.select_dtypes(include=['object']).columns

# One-hot encode Cat cols
dfencode = pd.get_dummies(df, columns=catcols)

# Print the encoded DataFrame
dfencode[0:10]

# Choosing the model
from sklearn.linear_model import LinearRegression

# The inputs
X1 = dfencode.drop(['attend', ], axis=1)

# Target variable
y1 = dfencode['attend']

# The model
model = LinearRegression()

# Fitting the model
model.fit(X1, y1)

# Finding the coefficients of the modek
Coeff1 = pd.Series(model.coef_, index=X1.columns)

# Print coefficients 
print(Coeff1.sort_values(ascending=False)[0:10])

# Import visuals
import matplotlib.pyplot as plt

# Make predictions on Inputs
ypred1 = model.predict(X1)

# Plot the predicted values against the actual values
plt.scatter(y1, ypred1)
sns.regplot(x=y1, y=ypred1, line_kws={'color': 'green'})
plt.xlabel('Actual Attendance v1')
plt.ylabel('Predicted Attendance v1')
plt.show()

# Finding the first versions R squared
from sklearn.metrics import r2_score
r2_score(y1, ypred1)*100

# Import regrex
import re

# Drop the opp column
coldrop = dfencode.filter(regex='^oppon').columns.tolist()

# Drop columns
Xdfen = dfencode.drop(['attend'] + coldrop, axis=1)

# Second variables
X2 = Xdfen
y2 = dfencode['attend']

model = LinearRegression()

# Fitting model using new variables
model.fit(X2, y2)

# Second round of coeffs
Coeff2 = pd.Series(model.coef_, index=X2.columns)

# Print the coefficients 
print(Coeff2.sort_values(ascending=False)[0:10])

# Make predictions on Inputs
ypred2 = model.predict(X2)

# Plot the predicted values against the actual values
plt.scatter(y2, ypred2)
sns.regplot(x=y2, y=ypred2, line_kws={'color': 'red'})
plt.xlabel('Actual Attendance v2')
plt.ylabel('Predicted Attendance v2')
plt.show()

# Finding the second versions R squared
r2_score(y2, ypred2)*100
