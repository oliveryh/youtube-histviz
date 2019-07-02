#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

# Imports
import spacy
from collections import defaultdict, Counter
import itertools
import plotly.figure_factory as ff
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.offline as py
import operator
import functools
import datetime as dt
import pandas as pd
import string
import json
import nltk
from nltk.corpus import stopwords


# In[64]:


# Open History File as dataframe
file = 'watch-history.json'
with open(file, encoding='utf8') as wh_file:
    wh_dict = json.load(wh_file)
    
wh = pd.DataFrame.from_dict(wh_dict)


# In[65]:


wh.head()


# In[66]:


# Drop columns except title and time
wh = wh[['title', 'time']]

# Remove "Watched" from title "Watched Cat Video" -> "Cat Video"
wh['title'] = wh['title'].apply(lambda x: x[7:])

# Lower Case "Cat Video" -> "cat video"
wh['title'] = wh['title'].apply(lambda x: x.lower())

# Remove Deleted videos
wh = wh.drop(wh[wh['title'].str.startswith('https://www.youtube.com')].index)

# Remove whitespace
wh['title'] = wh['title'].apply(lambda x: x.strip())
wh['title'] = wh['title'].apply(
    lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))

wh.head()


# In[67]:


nltk.download('stopwords')
stop_words = stopwords.words('english')

def find_ngrams(input_list, n):
    if n > 1:
        return zip(*(input_list[i:] for i in range(n)))
    else:
        return input_list

# Split words and remove stopwords
wh['unigrams'] = wh['title'].apply(lambda x: [word for word in x.split(' ') if word not in stop_words and word != ''])
wh['bigrams'] = wh['unigrams'].apply(lambda x: list(find_ngrams(x, 2)))
wh.head()


# In[68]:


# Create N-Gram Counters
def get_counter_from_column(df, column_name):
    ct = Counter()
    for row in df[column_name]:
        for element in row:
            ct[element] += 1
    return ct

bag_1 = get_counter_from_column(wh, 'unigrams')
bag_2 = get_counter_from_column(wh, 'bigrams')

bag_1.most_common(10)


# In[69]:


# Pick singular or plural for most common 500 words
def replace_count(counter, pair):
    removed, added = pair
    counter[added] += counter[removed]
    del counter[removed]
    wh['unigrams'] = wh['unigrams'].apply(lambda x: [added if unigram == removed else unigram for unigram in x])
    
remove_words = []
for word in bag_1.most_common(10):
    word = word[0]
    if word.endswith('s'):
        singular = word[:-1]
        plural = word
    else:
        singular = word
        plural = word + 's'
    if plural in bag_1 and singular in bag_1:
        if bag_1[plural] >= bag_1[singular]:
            remove_words.append((singular, plural))
        else:
            remove_words.append((plural, singular))

for removals in remove_words:
    replace_count(bag_1, removals)
    
bag_1.most_common(10)


# In[19]:


THRESHOLD = 0.3
for bigram in bag_2.most_common(1000):
    # print("{}: {}".format(bigram[0][0], bag_1[bigram[0][0]] * 0.75))
    if (bag_1[bigram[0][0]] * THRESHOLD) <= bag_2[bigram[0]]:
        del bag_1[bigram[0][0]]
        if (bag_1[bigram[0][1]] * THRESHOLD) <= bag_2[bigram[0]]:
            del bag_1[bigram[0][1]]
    else:
        del bag_2[bigram[0]]
print(bag_1[('video',)])
wh['ngrams'] = wh['unigrams'] + wh['bigrams']
bag_1_2 = (bag_1 + bag_2)


wh['time'] = wh['time'].apply(
    lambda x: dt.datetime.strptime(
        x.split('T')[0], '%Y-%m-%d'))


def avg_datetime(series):
    dt_min = series.min()
    deltas = [(x - dt_min).days for x in series]
    if len(deltas) == 0:
        print(series)
    return dt_min +         timedelta(functools.reduce(operator.add, deltas) // len(deltas))


def median_datetime(series):
    dt_min = series.min()
    deltas = [(x - dt_min).days for x in series]
    return dt_min + timedelta(days=deltas[len(deltas) // 2])


NUM_KEYWORDS = 350
palette = [
    'darkturquoise',
    'darkorange',
    'darkorchid',
    'mediumseagreen',
    'royalblue',
    'saddlebrown',
    'tomato']
plotly_colors = [palette[random.randrange(
    0, len(palette))] for i in range(NUM_KEYWORDS)]

# group_labels = ['minecraft', 'calisthenics', 'sex', 'dating', 'brexit', 'fail', 'minimalist', 'compilation', 'london', 'vegan', 'world', 'trump', 'tinder', 'react', 'flutter', 'summer', 'gopro']
removals = ["video", "trailer", "new", "best", "official", "removed",
            "music", "ft", "feat"] + [str(x) for x in list(range(100))]
removals += [("official", "video"), ("official", "trailer"),
             ("music", "video"), ("official", "music")]
group_labels = list(
    set([x[0] for x in bag_1_2.most_common(NUM_KEYWORDS)]) - set(removals))
data = pd.DataFrame([], columns=["keyword", "x", "y", "freq"])
for keyword in group_labels:
    dates = wh[wh['ngrams'].apply(lambda x: (
        True if keyword in x else False))]['time']
    data = data.append({
        "keyword": keyword if isinstance(keyword, str) else " ".join(keyword),
        "x": avg_datetime(dates),
        "y": 0,
        "freq": len(dates)
    }, ignore_index=True)
for year in [2014, 2015, 2016, 2017, 2018, 2019]:
    for month in [1, 4, 7, 10]:
        selected_year = data['x'].apply(
            lambda x: (
                True if datetime(
                    year,
                    month,
                    1) < x <= datetime(
                    year,
                    month,
                    1) +
                timedelta(
                    days=90) else False))
        num_selected = len(data[selected_year == True])
        print(num_selected)
        print(data[selected_year]['keyword'])
        ys = [x + 1 / (num_selected + 1) + random.uniform(-0.02, +0.02)
              for x in np.linspace(0, 1, num_selected + 2)][:-2]
        random.shuffle(ys)
        data.loc[selected_year, 'y'] = ys


# In[ ]:




