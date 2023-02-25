#!/usr/bin/env python
# coding: utf-8

# #                                         Lets Grow More 
# 
# ##                         Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                               Name - Samiksha Makhija
# 
# # 
# 
# ## Task 3 - Music Recommendation 
# 
# ### Task Description - Music recommender system can suggest songs to users based on their listening pattern.
# 
# ### Dataset -  https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
# 

# ## 
# ## Importing neccessary Libraries / Packages

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import Recommenders as Recommenders


# ## Loading music data

# In[2]:


members = pd.read_csv(r'c:\LGMVIP\members.csv',parse_dates=["registration_init_time","expiration_date"])
members.head()


# In[3]:


df_train = pd.read_csv(r'c:\LGMVIP\train.csv')
df_train.head()


# In[5]:


df_songs = pd.read_csv(r'c:\LGMVIP\songs.csv')
df_songs.head()


# In[6]:


df_songs_extra = pd.read_csv(r'c:\LGMVIP\song_extra_info.csv')
df_songs_extra.head()


# In[7]:


df_test = pd.read_csv(r'c:\LGMVIP\test.csv')
df_test.head()


# ## Creating a new data

# In[8]:


res = df_train.merge(df_songs[['song_id','song_length','genre_ids','artist_name','language']], on=['song_id'], how='left')
res.head()


# In[9]:


train = res.merge(df_songs_extra,on=['song_id'],how = 'left')
train.head()


# In[10]:


song_id = train.loc[:,["name","target"]]
song1 = song_id.groupby(["name"],as_index=False).count().rename(columns = {"target":"listen_count"})


# In[11]:


song1.head()


# In[12]:


dataset=train.merge(song1,on=['name'],how= 'left')


# In[13]:


df=pd.DataFrame(dataset)


# In[14]:


df.drop(columns=['source_system_tab','source_screen_name','source_type','target','isrc'],axis=1,inplace=True)
df=df.rename(columns={'msno':'user_id'})


# ## Loading new dataset

# In[15]:


df.head()


# ## Data Preprocessing

# In[16]:


df.shape


# In[17]:


#checking null values
df.isnull().sum()


# In[18]:


#filling null values
df['song_length'].fillna('0',inplace=True)
df['genre_ids'].fillna('0',inplace=True)
df['artist_name'].fillna('none',inplace=True)
df['language'].fillna('0',inplace=True)
df['name'].fillna('none',inplace=True)
df['listen_count'].fillna('0',inplace=True)


# In[19]:


#Rechecking null values
df.isnull().sum()


# In[20]:


print("Total no of songs:",len(df))


# ## Creating a subset of the dataset

# In[21]:


df = df.head(10000)

#Merge song title and artist_name columns to make a new column
df['song'] = df['name'].map(str) + " - " + df['artist_name']


# ## Showing the most popular songs in the dataset

# The column listen_count denotes the no of times the song has been listened.Using this column, we’ll find the dataframe consisting of popular songs:

# In[22]:


song_gr = df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_gr['listen_count'].sum()
song_gr['percentage']  = song_gr['listen_count'].div(grouped_sum)*100
song_gr.sort_values(['listen_count', 'song'], ascending = [0,1])


#  ## Counting the number of unique users in the dataset

# In[23]:


users = df['user_id'].unique()
print("The no. of unique users:", len(users))


# Now, we define a dataframe train which will create a song recommender

#  ## Counting the number of unique songs in the dataset

# In[24]:


###Fill in the code here
songs = df['song'].unique()
len(songs)


# # Creating a song Recommender

# In[25]:


train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
print(train.head(5))


# ## Creating Popularity Based Music Recommendations
# 
# Using  popularity_recommender class we made in Recommenders.py package, we create the list given below:

# In[26]:


pm = Recommenders.popularity_recommender_py()                               #create an instance of the class
pm.create(train_data, 'user_id', 'song')

user_id1 = users[5]                                                          #Recommended songs list for a user
pm.recommend(user_id1)


# In[27]:


user_id2 = users[45]
pm.recommend(user_id2)


# ## Building a Song Recommender with Personalization
# 
# We are now creating an item similarity based collaborative filtering model that allows us to make personalized recommendations to each user. 

# ## Creating Similarity Based Music Recommendation

# In[37]:


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')



# ## Using the Personalized Model to make some song recommendations

# In[38]:


#Print the songs for the user in training data
user_id1 = users[1]
#Fill in the code here
user_items2 = is_model.get_user_items(user_id2)
print("------------------------------------------------------------------------------------")
print("Songs played by second user %s:" % user_id2)
print("------------------------------------------------------------------------------------")

for user_item in user_items1:
    print(user_item)

print("----------------------------------------------------------------------")
print("Similar songs recommended for the second user:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id1)


# In[39]:


user_id2 = users[7]
#Fill in the code here
user_items2 = is_model.get_user_items(user_id2)
print("------------------------------------------------------------------------------------")
print("Songs played by second user %s:" % user_id2)
print("------------------------------------------------------------------------------------")

for user_item in user_items2:
    print(user_item)

print("----------------------------------------------------------------------")
print("Similar songs recommended for the second user:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id2)


# The lists of both the users in popularity based recommendation is the same but different in case of similarity-based recommendation. 

# ### We can also use the model to find similar songs to any song in the dataset

# In[40]:


is_model.get_similar_items(['U Smile - Justin Bieber'])


# # THANK YOU !
