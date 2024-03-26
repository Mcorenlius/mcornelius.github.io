import pandas as pd
import numpy as np
import xlrd
import os

os.getcwd()

moviesdf = pd.read_csv('movies.csv')
ratingsdf = pd.read_csv('ratings.csv')
tagsdf = pd.read_csv('tags.csv')

moviesdf.head()

ratingsdf.head()

tagsdf.head()

ratingsdf = ratingsdf.merge(moviesdf,on='movieId', how='left')
ratingsdf.head()

Avgratings = pd.DataFrame(ratingsdf.groupby('title')['rating'].mean())
Avgratings.head()

Avgratings['Total Ratings'] = pd.DataFrame(ratingsdf.groupby('title')['rating'].count())
Avgratings.head()

muser = ratingsdf.pivot_table(index='userId',columns='title',values='rating')
muser.head()

list(moviesdf['title'].unique())

from warnings import filterwarnings
filterwarnings('ignore')

# Change title here to get new list each time
corr = muser.corrwith(muser['Home Alone (1990)'])
corr.head()

recom = pd.DataFrame(corr,columns=['Correlation'])
recom.dropna(inplace=True)
recom = recom.join(Avgratings['Total Ratings'])
recom.head()

recc = recom[recom['Total Ratings']>100].sort_values('Correlation',ascending=False).reset_index()

recc = recc.merge(moviesdf,on='title', how='left')
recc.head()

recc[recc['Correlation'] >= 0.40]
