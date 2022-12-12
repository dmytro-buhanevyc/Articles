
import csv
import datetime
import glob
import inspect
import io
import json  # library to handle JSON files
import os
import re
import shutil
import sys
import textwrap
import time
import urllib.request
import winsound
from collections import Counter, OrderedDict
from datetime import date, datetime
from hashlib import new
from netrc import netrc
from pathlib import Path
from urllib import response
from urllib.error import URLError
import branca.colormap as cm
import datapane as dp
import en_core_web_sm
import folium  # map rendering library
import geopandas as gpd
import jsonpickle
import nltk
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import pytextrank
import pytorch_lightning as pl
import requests
import scattertext as st
import seaborn as sns
import snscrape
import snscrape.base
import snscrape.modules.facebook as facebookscraper
import snscrape.modules.twitter as sntwitter
import snscrape.modules.twitter as twitterScraper
import spacy
import spacy.attrs
import streamlit as st
import torch
import tweepy
from bs4 import BeautifulSoup
from dframcy import DframCy
from flashtext import KeywordProcessor
from folium.features import GeoJsonPopup, GeoJsonTooltip
from geopy.geocoders import Nominatim
from googletrans import Translator
from IPython.display import display
from ipywidgets import HBox, VBox, interactive, widgets
from matplotlib import pyplot as plt
from newspaper import Article
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from pandas_profiling import ProfileReport
from plotly.subplots import make_subplots
from prophet import Prophet
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from spacy import displacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from textblob import TextBlob
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from plotly_calplot import calplot
from itertools import zip_longest
from itertools import accumulate

from tqdm import tqdm
tqdm.pandas()
!datapane login - -token = 8a2c9de9f8d634b9a4162974d4ba685dfaf82708
duration = 1000  # milliseconds
freq = 440  # Hz
directory = os.getcwd()

#CONGRESS DATA
# downloaded from https://github.com/unitedstates/congress-legislators


congress117 = pd.read_json(r"https://theunitedstates.io/congress-legislators/legislators-current.json")
#maing all terms into a new list
df1 = congress117['terms'].apply(pd.Series)
df1 = df1.ffill(axis=1).iloc[:, -1]
#rename last column to 'latest_term'
df1.rename('latest_term', inplace=True)
congress117 = pd.concat([congress117, df1], axis=1)



#convert 'latest_term' column to string
congress117['latest_term'] = congress117['latest_term'].astype(str)
#extract 'party' from 'latest_term' column into a new Column
congress117['party'] = congress117['latest_term'].str.extract(r" 'party': '(\w+)'")
#extract 'state' from 'latest_term' column into a new Column
congress117['state'] = congress117['latest_term'].str.extract(r" 'state': '(\w+)'")
#extract 'district' from 'latest_term' column into a new Column
congress117['district'] = congress117['latest_term'].str.extract(r" 'district': (\w+)")


congress117['type']=congress117['latest_term'].str.extract(r"'type': '(\w+)'")
#extract 'start' from 'latest_term' column into a new Column
#extract date from 'latest_term' column into a new Column
congress117['start'] = congress117['latest_term'].str.extract(r"'start': '(\d+-\d+-\d+)'")
congress117['end'] = congress117['latest_term'].str.extract(r"'end': '(\d+-\d+-\d+)'")

congress117.party.value_counts()





def search(df: pd.DataFrame, substring: str, case: bool = False) -> pd.DataFrame:
    found_item = np.column_stack([df[col].astype(str).str.contains(
        substring.lower(), case=case, na=False) for col in df])
    return df.loc[found_item.any(axis=1)]
search(congress117, 'schatz')





# SOCIAL MEDIA ### congress 117 has 538 rows, but social media only 528 and 2 are Null


congress117_socialmedia = pd.read_json(r"https://theunitedstates.io/congress-legislators/legislators-social-media.json ")
#convert social to string
congress117_socialmedia['social'] = congress117_socialmedia['social'].astype(str)
congress117_socialmedia['twitter']=congress117_socialmedia['social'].str.extract(r"'twitter': '(\w+)'")
#find rows where twitter is null / there are 3 null values, overall handles - 525

congress117_socialmedia[congress117_socialmedia['twitter'].isnull()]
congress117_socialmedia['twitter'].count()

# making a list of twitter handles to scrape
congress117_twitter_list = congress117_socialmedia['twitter'].tolist()



####### MAKING A SAMPLE

congress117_socialmedia_sample = congress117_socialmedia.sample(n=11)

congress117_twitter_list = congress117_socialmedia_sample['twitter'].tolist()

length_to_split = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,]
master_list =  [congress117_twitter_list[x - y: x] for x, y in zip(accumulate(length_to_split), length_to_split)]

#drop empty lists
master_list = [x for x in master_list if x != []]




#
split = np.array_split(congress117_twitter_list, 53)

print(split[0])
#

# find unique list in the total dataframe
#loop in sequence over list of lists

congress117tweets_updated = []
users_name = split[0]

users_name = master_list

for n, k in enumerate(users_name):
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(' from:{} since:2017-01-03'.format(users_name[n])).get_items()):
        if i > 10:
            break
        congress117tweets_updated.append({"id": tweet.id, "Username": tweet.user.username, "date": tweet.date, "link": tweet.url,
                                   "content": tweet.content, "Likes": tweet.likeCount, "Retweets": tweet.retweetCount,
                                   "Quoted": tweet.quoteCount, 'Language': tweet.lang, 'Replied_to': tweet.inReplyToUser, "Mentioned": tweet.mentionedUsers,
                                   "Hashtags": tweet.hashtags,
                                   "Replies": tweet.replyCount, "userdescription": tweet.user.description, "Name": tweet.user.displayname,
                                   "Verified": tweet.user.verified,
                                   "Followers": tweet.user.followersCount,
                                   "Friends": tweet.user.friendsCount,
                                   "Location": tweet.user.location})


f = open("congress117tweets_updated.json", "w")
j = json.dumps(congress117tweets_updated, indent=4, sort_keys=True, default=str)
f.write(j)
f.close()
if os.stat("congress117tweets_updated.json").st_size <= 5:
    print('No Tweets found')
if os.stat("congress117tweets_updated.json").st_size >= 5:
    print('Completed')
    winsound.PlaySound(r'C:\Users\dbukhanevych\Downloads/mixkit-dog-barking-twice-1.wav',
                       winsound.SND_FILENAME | winsound.SND_ASYNC)

#conver json to pandas
congress117tweets_updated = pd.read_json(r"C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\congress117tweets_updated.json")

### LOCAL OPTION ###
datetime.datetime(2001, 5, 1)
now = str(datetime.datetime.now())[:19]
now = now.replace(":", "_")

src_file = os.path.join(directory, 'congress117tweets_updated.json')
dest_dir = directory+"/Tweets"+str(now)+".json"

shutil.copy(src_file, dest_dir)  # copy the file to destination dir


# renaming all #

def main():

    folder = r"C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\Tweets"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"tweets{str(count)}.json"
        # foldername/filename, if .py file is outside folder
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':

    # Calling main() function
    main()

# END

df = pd.read_json(
    r'C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\Tweets\tweets0.json')
df


jsv1 = os.listdir(
    r"C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\Tweets")
files = glob.glob(
    r"C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\Tweets\*")
json_dir = r'C:\Users\dbukhanevych\Anaconda3\envs\TCA\hello\Academic\NYU\Tweets'
json_pattern = os.path.join(json_dir, '*.json')
file_list = glob.glob(json_pattern)

dfs = []
for file in file_list:
    with open(file) as f:
        json_data = pd.json_normalize(json.loads(f.read()))
        json_data['site'] = file.rsplit("/", 1)[-1]
    dfs.append(json_data)
df = pd.concat(dfs)

df.Username.nunique()


result = df.to_json(orient="split")
df.to_json(r'congress_tweets_all.json', orient='split')



# unique 1.9 million tweets
df.id.nunique()


# duplicates
df[df.duplicated(['id'], keep=False)].sort_values("id")

###########################

# GUNS Lost - Lost bicycles - existing
#Quick search
#mass_shooting_tweets = search(df, 'RepAmata')
#ukraine_tweets = search(df, 'Ukraine')


#option2 better
Search_List = ['mass shooting', 'mass shootings', 'AR-15',
               'guns', 'firearm', 'firearms', 'school shooting']
mass_shooting_tweets = df[df['content'].str.contains(
    '|'.join(Search_List), case=False)]


###### COMBINING ALL DATASETS
df.Name.nunique()
mass_shooting_tweets.date.max()


df_congress_terms
df_congress_terms = df_congress_terms.astype(str)
df_congress_terms['id'] = df_congress_terms['id'].astype(str)


congress117_social_handles
congress117_social_handles['id1'] = congress117_social_handles['id'].astype(str)
congress117_social_handles = congress117_social_handles.astype(str)


###############3

########### extracting buiguide

df_congress_terms = df_congress_terms.assign(bioguide=df_congress_terms['id'].str.extract('(bioguide.{11})'))
df_congress_terms['bioguide'] = df_congress_terms['bioguide'].str.replace('\W', '')

congress117_social_handles = congress117_social_handles.assign(bioguide=congress117_social_handles['id1'].str.extract('(bioguide.{11})'))
congress117_social_handles['bioguide'] = congress117_social_handles['bioguide'].str.replace('\W', '')


##### 8 members don't have an official account, data is only for *Current social media accounts for Members of Congress. Official accounts only (no campaign or personal accounts).*
# 

search(mass_shooting_tweets,'titus')

congress117_social_handles.to_csv('Test.csv')


### combining the datasets

#merge datasets on bioguide USING COPILOT / NOT SURE THIS IS RIGHT

df_congress_terms = pd.merge(df_congress_terms, congress117_social_handles, on='bioguide', how='left')





# Combining final 2 datasets - all legislators data, social media data + actual tweets
mass_shooting_tweets = mass_shooting_tweets.reset_index()

test = df_congress_terms.dropna(subset=['twitter'])

mass_shooting_tweets['party']=mass_shooting_tweets['Username'].map(dict(zip(df_congress_terms['twitter'], df_congress_terms['party'])))
mass_shooting_tweets=mass_shooting_tweets.drop(columns='site')
mass_shooting_tweets['party'] = mass_shooting_tweets['party'].str.replace('\W', '')
mass_shooting_tweets['party'] = mass_shooting_tweets['party'].str.replace('party', '')



mass_shooting_tweets.Username.nunique()


################################### Aggregating by party 117 CONGRESS TIME LINE

#######################3
###################
#########


#group by party and createa a pie chart
mass_shooting_tweets_correct_dates = mass_shooting_tweets[mass_shooting_tweets['date'] >= '2021-01-03'].reset_index()
mass_shooting_tweets_correct_dates= mass_shooting_tweets_correct_dates.drop_duplicates(subset=['id'], keep='first')

#group by party 
test = mass_shooting_tweets_correct_dates.groupby('party').size()
#turn to dataframe with columns named Party and Count
test = test.to_frame(name = 'Count').reset_index()
#normalize to % wit 2 decimals and create a new column
test['%'] = test['Count'].apply(lambda x: round(x/test['Count'].sum()*100,2))
#create a plotly pie chart with party and %, make Democrats blue, Republicans red and Independents grey

fig = px.pie(test, values='%', names='party',title='Tweets related to mass shootings by Party')
#change pie colors to blue, red and grey
fig.update_traces(marker=dict(colors=['rgb(49,130,189)', 'rgb(189,189,189)', '#b22234', ]))
fig.update_layout(width = 500, height = 500)
fig.show()






# convert date to datetime and normalize by day
mass_shooting_tweets['date'] = pd.to_datetime(mass_shooting_tweets['date'])
mass_shooting_tweets['date'] = mass_shooting_tweets['date'].dt.normalize()

mass_shooting_tweets= mass_shooting_tweets.drop_duplicates(subset=['id'], keep='first')


#count values per party by date
test = mass_shooting_tweets.groupby(['party', 'date']).size().reset_index(name='Count')

#drop every row before January 3, 2021
test = test[test['date'] >= '2021-01-03']



#plot line chart and make colors blue for Democrat, grey for Independent and red for Republican
fig = px.line(test, x="date", y="Count", color='party', title='Tweets related to mass shootings by Party and date',
color_discrete_map={'Democrat': 'rgb(49,130,189)', 'Independent': 'rgb(189,189,189)', 'Republican': '#b22234'})
fig.update_layout(width = 1000, height = 500)

fig.show()

#calculate average daily number of tweets per party
daily_mean_mass_shooting = test.groupby('party')['Count'].mean()


# LOOKING AT THE WHOLE NUMBER OF TWEETS

df_full_test = df
df_full_test = df_full_test[df_full_test['date'] >= '2021-01-03']

df_full_test['party']=df_full_test['Username'].map(dict(zip(df_congress_terms['twitter'], df_congress_terms['party'])))
df_full_test=df_full_test.drop(columns='site')
df_full_test['party'] = df_full_test['party'].str.replace('\W', '')
df_full_test['party'] = df_full_test['party'].str.replace('party', '')

#group by 
df_full_test['date'] = pd.to_datetime(df_full_test['date'])
df_full_test['date'] = df_full_test['date'].dt.normalize()

test2 = df_full_test.groupby(['party', 'date']).size().reset_index(name='Count')
#find daily average number of tweets per party
daily_mean_mass_shooting_total = test2.groupby('party')['Count'].mean()

#merge the 2 datasets
daily_mean_mass_shooting_total = pd.merge(daily_mean_mass_shooting_total, daily_mean_mass_shooting, on='party', how='left')
daily_mean_mass_shooting_total = daily_mean_mass_shooting_total.rename(columns={'Count_x': 'Total tweets', 'Count_y': 'Mass shooting tweets'})
daily_mean_mass_shooting_total = daily_mean_mass_shooting_total.reset_index()

#find ratio of mass shooting tweets to total tweets
daily_mean_mass_shooting_total['Ratio'] = daily_mean_mass_shooting_total['Mass shooting tweets']/daily_mean_mass_shooting_total['Total tweets']


#plot line chart with Total tweets, Mass shooting tweets and Ratio (If tweet ID matchers - )
#mass_shooting_tweets_correct_dates and df_full_test

#add a dummy column to the df_full_test dataset if the tweet ID matches the mass_shooting_tweets_correct_dates dataset
df_full_test['Mass_shooting_tweet'] = np.where(df_full_test['id'].isin(mass_shooting_tweets_correct_dates['id']), 'Yes', 'No')

df_full_test['Mass_shooting_tweet'].value_counts()
#find duplicate id and sort by id

#drop duplicates
df_full_test = df_full_test.drop_duplicates(subset=['id'], keep='first').reset_index()

#group by 'Mass shooting tweet', date and party
test3 = df_full_test.groupby(['Mass_shooting_tweet', 'date', 'party']).size().reset_index(name='Count')

#create a new column with ratio of Yes to No per party by date
test3['Ratio'] = test3.groupby(['date', 'party'])['Count'].apply(lambda x: x/x.sum())

#plot Ration per party and date
#keep Independent
test_all_parties = test3[test3['Mass_shooting_tweet'] == 'Yes']


test_dem= test3[test3['party'] == 'Democrat']

test_dem = test_dem[test_dem['Mass_shooting_tweet'] == 'Yes']

# plot ratio line per day with facet

fig = px.line(test_all_parties, x="date", y="Ratio", color='party', facet_col = 'party', title='Ratio of tweets related to mass shootings by Party and date',
color_discrete_map={'Democrat': 'rgb(49,130,189)', 'Independent': 'rgb(189,189,189)', 'Republican': '#b22234'})
fig.update_layout(width = 1000, height = 500)
fig

#create mekko chart with total tweets, mass shooting tweets and ratio



#find all values for April, 11, 2021 to test ratio
testing = df_full_test[df_full_test['date'] == '2021-12-14']

testing= testing[testing['party'] == 'Democrat']

#count Mass shooting values per day
test4 = testing.groupby(['Mass_shooting_tweet']).size().reset_index(name='Count')
test4







###############33
#END  # E ## N ## D 
##############


# END SEARCH
df_reset = df.reset_index(drop=True)
search(df_reset, 'mass shooting')

# WORkING SHOW AND TELLs


mass_shooting_tweets['date'] = pd.to_datetime(
    mass_shooting_tweets['date']).dt.normalize()
mass_shooting_tweets_count = mass_shooting_tweets.groupby(
    ['date']).size().reset_index(name='counts')

mass_shooting_tweets_count['counts'].sum()


# line
colors = ['#b22234','rgb(49,130,189)',  'rgb(67,67,67)', 'rgb(189,189,189)']

###############################
#test



#mass_shooting_tweets = mass_shooting_tweets.dropna(subset=['party'])




########################

mass_shooting_tweets_count = mass_shooting_tweets.groupby(
    ['date', 'party']).size().reset_index(name='counts')
mass_shooting_tweets.to_csv('Test.csv')



fig1 = px.line(mass_shooting_tweets_count, x='date',
              y='counts', color = 'party',  facet_col='party', facet_col_wrap=3, color_discrete_sequence=colors)
today = date.today()
fig1.update_layout(width=1200, height=500,
                  title="Tweets relating to gun violence by Members of the 117 Congress <br><sup></sup><br>",
                  xaxis_title="Date",
                  legend_title="Party",
                  # plot_bgcolor='white'
                  )
for axis in fig1.layout:
    if type(fig1.layout[axis]) == go.layout.YAxis:
        fig1.layout[axis].title.text = ''
    if type(fig1.layout[axis]) == go.layout.XAxis:
        fig1.layout[axis].title.text = ''
fig1.update_layout(
    # keep the original annotations and add a list of new annotations:
    annotations = list(fig1.layout.annotations) + 
    [go.layout.Annotation(
            x=-0.07,
            y=0.5,
            font=dict(
                size=14
            ),
            showarrow=False,
            text="Total number of tweets",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    ]
)
fig1.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.14, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig1.show()


# individual tweets


cols = mass_shooting_tweets.columns.tolist()
mass_shooting_tweets = mass_shooting_tweets[['id', 'Name', 'userdescription', 'Username', 'content', 'date', 'Likes', 'Retweets', 'Quoted', 'Replied_to', 'Replies', 'Verified', 'Followers', 'Friends',
                                             'Hashtags', 'Location', 'Mentioned', 'Language', 'link', 'party']]
mass_shooting_tweets.userdescription = mass_shooting_tweets.userdescription.str.wrap(
    60)
mass_shooting_tweets.userdescription = mass_shooting_tweets.userdescription.apply(
    lambda x: x.replace('\n', '<br>'))
mass_shooting_tweets.content = mass_shooting_tweets.content.str.wrap(
    60)
mass_shooting_tweets.content = mass_shooting_tweets.content.apply(
    lambda x: x.replace('\n', '<br>'))




colors = ['#b22234', 'rgb(49,130,189)',  'rgb(189,189,189)',  'rgb(67,67,67)']

fig2 = px.scatter(mass_shooting_tweets, x="date", y="Likes", size='Followers',
                 color="party", color_discrete_sequence=colors,
                 custom_data=["Username", 'userdescription',
                              'date', 'content', 'Followers', 'Replies'],
                 log_y=False, size_max=20)
fig2.update_traces(
    hovertemplate="<br>".join([
        "Username: %{customdata[0]}",
        "userdescription: %{customdata[1]}",
        "Date: %{customdata[2]}",
        "Content: %{customdata[3]}",
        "Followers: %{customdata[4]}",
        "Replies: %{customdata[5]}",
    ])
)
fig2.update_layout(width=900, height=600,
                  title="Tweets relating to gun violence by Members of the 117 Congress<br><sup> Horizontal line - Uvalde, TX shooting</sup>",
                  xaxis_title="Date",
                  yaxis_title="Likes",
                  legend_title="Name",
                  )
today = date.today()
fig2.update_layout(
    hoverlabel=dict(
        font_size=11,
        font_family="Gill Sans, serif"
    )
)
fig2.add_vline(x='2022-05-24 15:30:01+00:00', line_width=0.4,
              line_dash="dash", fillcolor="red", opacity=0.8)
fig2.update_layout(
    xaxis_range=['2018-05-24 15:30:01+00:00', '2022-10-24 15:30:01+00:00'])

fig2.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.23, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig2.show()

###########################3 density

mass_shooting_tweets_count = mass_shooting_tweets.groupby(
    ['date', 'party']).size().reset_index(name='counts')

mass_shooting_tweets_count['month_year'] = mass_shooting_tweets_count['date'].dt.to_period('M')



mass_shooting_tweets.Name.unique()
Search_List = ['John Kennedy', 'Mark Warner', 'Rep. Liz Cheney',
               'Richard Blumenthal', 'Congressman Raja Krishnamoorthi', 'Rep. Pete Aguilar', 'Rep. Ted Lieu', 'Rep. Matt Gaetz',
              'Mo Brooks', 'Del. Kilili Sablan']
mass_shooting_tweets_count = mass_shooting_tweets_count[mass_shooting_tweets_count['Name'].str.contains(
    '|'.join(Search_List), case=False)]

###################################

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list('ABCDEFGHIJ'), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "x")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)










############## END DENSITY 




#seach list###

Search_List = ['taira', 'Tatsuro', 'tatsuro',
               'UFC', 'MMA', 'Mokaev', 'mokaev', 'Fate']
german_public_clean = german_public_clean[~german_public_clean['content'].str.contains(
    '|'.join(Search_List), case=True)]


#####
df = df.rename(columns={'Username': 'twitter'})

df_total = pd.merge(df, congress117_social_handles, on="twitter", how='left')


####################


# end


# exploding full dataset with ID
df_total['date'].min()


df_total.to_json('congress117tweets.json', orient='records', lines=True)

congress117_social_handles
congress117.terms.iloc[1]


congress117_social_handles = pd.concat([congress117_socialmedia, df2], axis=1)

########### TOTAL LINE ###
df_total['date'] = pd.to_datetime(
    df_total['date']).dt.normalize()


df_total


df3 = df_total['id_y'].apply(pd.Series)


fig = px.scatter(df_total, x='date', y='Likes', color='Name')
today = date.today()
fig.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.23, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig.show()


# TWITTER TRIDENT

background_checks = pd.ExcelFile(r'data/Background checks.xlsx')

df1 = pd.read_excel(background_checks, 'Monthly')

daily = pd.read_excel(background_checks, 'Daily')
df1['Year'] = pd.to_datetime(
    df1['Year']).dt.normalize()

df1['month'] = pd.DatetimeIndex(df1['Year']).month
df1['year'] = pd.DatetimeIndex(df1['Year']).year


fig = px.line(df1, x='Year', y=['New York', 'South Dakota', 'North Dakota'])
today = date.today()
fig.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.23, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig.show()


df1_matrix = df1.pivot('month', 'year', 'Colorado')


fig = px.imshow(df1_matrix, color_continuous_scale='BuPu', text_auto=True,
                labels=dict(color='Background Checks',
                            zmin='1', zmax='12')
                )

fig.update_layout(width=800, height=800,
                  title="Monthly background checks<br><sup>Colorado state</sup>",
                  xaxis_title="Year",
                  yaxis_title="Month",
                  legend_title="None`",
                  plot_bgcolor='white'
                  )
fig.update_layout(
    yaxis=dict(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tickfont=dict(size=8),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', "Sep", 'Oct', 'Nov', 'Dec'])
)
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickfont=dict(size=8),
    )
)
today = date.today()
fig.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.17, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)

# fig.update_traces(customdata=customdata, hovertemplate=hovertemplate)
fig.show()


mass_shootings = pd.read_excel(r'Mass Shootings (08_1966-04_2021) (1).xlsx')


mass_shootings_colorado = mass_shootings[mass_shootings['State'] == 'Colorado']


mass_shootings['State'].value_counts()

colors = ['rgb(49,130,189)', '#b22234', 'rgb(189,189,189)',  'rgb(67,67,67)']


fig2 = px.scatter(mass_shootings_colorado, x='Date', y='Fatalities', color='FameSeeker_lank', size='Fatalities',
                  color_discrete_sequence=colors, custom_data=[
                      "Event", 'City', 'State', 'Date', 'Target_Group', 'Shooting_Location', 'Level_of_Security', 'Fatalities'],
                  log_x=False, size_max=30)
fig2.update_traces(
    hovertemplate="<br>".join([
        "Event: %{customdata[0]}",
        "City: %{customdata[1]}",
        "State: %{customdata[2]}",
        "Date: %{customdata[3]}",
        "Target_Group: %{customdata[4]}",
        "Shooting_Location: %{customdata[5]}",
        "Level_of_Security: %{customdata[6]}",
        "Fatalities: %{customdata[7]}",

    ])
)
fig2.update_layout(width=900, height=550,
                   title="Mass shootings in Colorado<br><sup>Fame seekers, fatalities</sup>",
                   xaxis_title="Date",
                   yaxis_title="Fatalities",
                   legend_title="Fame seekers",
                   plot_bgcolor='white'
                   )
today = date.today()
fig2.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.18, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig2.show()


# test

df = pd.DataFrame(
    index=pd.date_range("1-jan-1999", periods=22),
    data={c: np.random.randint(1, 10, 22) for c in list("AB")},
)

fig = make_subplots(rows=2, cols=1, row_width=[0.8, 0.2])

for t in px.line(df, y=df.columns).data:
    fig.add_trace(t, row=1, col=1)

for t in px.imshow(df1_matrix,  text_auto=True, labels=dict(color='Background Checks')).data:
    fig.add_trace(t, row=2, col=1)

fig.update_layout(width=800, height=800, coloraxis=dict(colorscale='BuPu'), showlegend=False,
                  title="Mass shootings in Colorado<br><sup>Fame seekers, fatalities</sup>",
                  xaxis_title="Yeary",
                  yaxis_title="Fatalities",
                  legend_title="Fame seekers",
                  plot_bgcolor='white'
                  )
today = date.today()
fig.add_annotation(
    text=(f"NYU Tandon School of Engineering | {today}<br>Authors: TBD"), showarrow=False, x=0, y=-0.18, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-1, yshift=-5, font=dict(size=10, color="lightgrey"), align="left",)
fig.update_traces(xaxis="x")
fig['layout']['yaxis2'].update(dict(
    tickmode='array',
    tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tickfont=dict(size=11), autorange="reversed",
    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', "Sep", 'Oct', 'Nov', 'Dec'])
)
fig['layout']['xaxis2'].update(dict(
    tickmode='array',
    tickfont=dict(size=8)))
fig.show()


fig.update_layout(
    coloraxis_colorbar={"len": 0.5, "yanchor": "bottom"},
    legend={"yanchor": "top", "y": 0.5, "tracegroupgap": 0},
)


# working

df = pd.DataFrame(
    index=pd.date_range("1-jan-2022", periods=15),
    data={c: np.random.randint(1, 10, 15) for c in list("AB")},
)

fig = make_subplots(rows=2, cols=1)
for t in px.line(df, y=df.columns).data:
    fig.add_trace(t, row=1, col=1)

for t in px.imshow(mass_shootings.values.T, x=df.index, y=df.columns).data:
    fig.add_trace(t, row=2, col=1)


fig.update_traces(xaxis="x")
fig.update_layout(
    coloraxis_colorbar={"len": 0.5, "yanchor": "bottom"},
    legend={"yanchor": "top", "y": 0.5, "tracegroupgap": 0},
)


# R  A  N  D  O  M

#### making all in one ##

dp.Report(
    dp.Page(title="Tweets on gun violence by US Congress",
            blocks=[
                dp.Plot(fig1),
                dp.Plot(fig2),

            ]),
    dp.Page(title="Dataset", blocks=[
        dp.DataTable(mass_shooting_tweets)
    ])
).upload(name='NYU Gun Violence Research Project')
