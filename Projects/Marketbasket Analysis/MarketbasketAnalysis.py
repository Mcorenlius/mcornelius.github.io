SQL Code
DECLARE @StartDate Date = '20100101'
DECLARE @EndDate Date = '20111231'
SELECT
T.Bill_Number
, S.Item_Name
, S.Quantity_Sold
, S.[Date]
, S.[Time]
, S.Price
, Customer_ID
, Country
FROM TRANSACTIONS T
INNER JOIN SALES S ON S.Bill_Number = T.Bill_Number
WHERE S.[Date] BETWEEN @StartDate AND @EndDate

Python Code
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns from mlxtend.frequent_patterns 
import apriori from mlxtend.frequent_patterns 
import association_rules 
import matplotlib.pyplot as plt

sns.set(style="darkgrid", color_codes=True) pd.set_option('display.max_columns', 75)

# Read the Excel data = pd.read_excel('MarketBasketAnalysis_Copy.xlsx') # Convert the 'Date' column data['Date'] = pd.to_datetime(data['Date']) # Find the unique dates unique_dates = data['Date'].unique() # Sort the unique dates unique_dates_sorted = sorted(unique_dates, reverse=True) # Select the last 7 unique dates last7daysdates = unique_dates_sorted[:7] # Filter the DataFrame data_filtered = data[data['Date'].isin(last7daysdates)] # Check info data_filtered.info()
data_filtered.head()

data_filtered.describe()

color = plt.cm.rainbow(np.linspace(0, 1, 40)) 
data_filtered["Item_Name"].value_counts().head(10).plot.bar(color = color, figsize=(8,5)) 
plt.title('Freq of most popular items', fontsize = 15) 
plt.xticks(rotation = 90 ) 
plt.grid() plt.show()

# Filter rows 
data_filtered2 = data_filtered.dropna(subset=['Item_Name']) 

# Group items 
transactions = data_filtered2.groupby('Bill_Number').agg({'Item_Name': list, 'Quantity_Sold': list}).reset_index() 

# Display the first few rows 
transactions.head()

# Aggregate item 
item_summary = data_filtered2.groupby('Item_Name')['Quantity_Sold'].sum().reset_index() 
# Sort 
item_summary_sorted = item_summary.sort_values(by='Quantity_Sold', ascending=False) 
# Display 
print(item_summary_sorted.head(25))

# Define target 
item target_item = "POPCORN HOLDER" 
dictionary item_frequencies = {} 
counter transaction_counter = 0 
# Iterate through each transaction 
for index, row in transactions.iterrows(): 
items = row['Item_Name'] 
if target_item in items: 
transaction_counter += 1 
# Increment the frequency 
for item in items: 
if item != target_item: 
if item in item_frequencies:
item_frequencies[item] += 1 
else: 
item_frequencies[item] = 1 
# Convert the item frequencies dictionary 
item_frequencies_df = pd.DataFrame(list(item_frequencies.items()), columns=['Item_Name', 'Frequency'])
# Sort DataFrame 
item_frequencies_df = item_frequencies_df.sort_values(by='Frequency', ascending=False)
# Calculate the support 
target_item_support = transaction_counter / len(transactions)
# Display results 
print("Target item:", target_item) 
print("Number of transactions found:", transaction_counter) 
print("Support of the target item:", target_item_support)
print("\nFrequency of items purchased alongside the target item:") print(item_frequencies_df.head(10))

# Get the first 10 items 
top_items = item_frequencies_df.head(10) 

# horizontal bar 
plot plt.figure(figsize=(10, 6)) 
plt.barh(top_items['Item_Name'], top_items['Frequency'], color='skyblue')
plt.xlabel('Frequency') 
plt.ylabel('Item Name') 
plt.title('Top 10 Items Purchased Alongside {}'.format(target_item)) 
plt.gca().invert_yaxis() 
# Invert y-axis to display items with highest frequency at the top 
plt.show()

# Calculate the likelihood (percentage) 
item_frequencies_df['Likelihood'] = (item_frequencies_df['Frequency'] / transaction_counter) * 100 
# Get the first 10 
top_items = item_frequencies_df.head(10) 
# Display the first 10 
print('Top 10 Items Purchased Alongside "{}" with Likelihood:'.format(target_item)) 
print(top_items)

# Visualize the likelihood 
plt.figure(figsize=(10, 6)) 
plt.barh(top_items['Item_Name'], top_items['Likelihood'], color='skyblue') 
plt.xlabel('Likelihood (%)') plt.ylabel('Item Name') plt.title('Likelihood of Items Purchased Alongside "{}"'.format(target_item)) 
plt.gca().invert_yaxis() 
plt.show()

# target item 
target_item = "POPCORN HOLDER" 
# flag
flag = False 
# Iterate through transactions 
for index, row in transactions.iterrows():
items = row['Item_Name'] quantities = row['Quantity_Sold'] 
if target_item in items: 
# Get index 
target_index = items.index(target_item) 
# Get quantity sold 
target_quantity = quantities[target_index]
# Check 
if target_quantity > 50: flag = True break
# Print flag value 
print(flag)
