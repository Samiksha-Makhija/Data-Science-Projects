#!/usr/bin/env python
# coding: utf-8

# #                                                  Lets Grow More 
# 
# ##                          Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                                      Name - Samiksha Makhija
# 
# 
# # 
# 
# ## Task 5 - Exploratory Data Analysis on Dataset - Terrorism
# 
# ### Task Description - As a security / defense analyst, try to find out the hot zone of terrorism (You can choose any of the tool of your choice like : Python / R / Tableau / PowerBI / Excel / SAP / SAS) 
# 
# 
# ### Exploratory Data Analysis - Exploratory Data Analysis (EDA) refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# 
# ### Dataset: https://bit.ly/2TK5Xn5

# ## 
# ## Importing all the important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# ## Loading dataset

# ### Read the CSV file

# In[2]:


dataset=pd.read_csv('globalterrorismdb.csv',encoding='ISO-8859-1',low_memory=False)
dataset.head()


# ### Shape of Dataset

# In[3]:


dataset.shape


# ### Dataset Columns

# In[4]:


dataset.columns


# ### Renaming the necessary columns

# In[5]:


dataset.rename(columns={'iyear':'Year','imonth':'Month','city':'City','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
dataset['Casualities'] = dataset.Killed + dataset.Wounded
dataset=dataset[['Year','Month','Day','Country','Region','City','latitude','longitude','AttackType','Killed','Wounded','Casualities','Target','Group','Target_type','Weapon_type']]
dataset.head(10)


# ### Checking for Missing data

# In[6]:


dataset.isnull().sum()


# ### Removing the Missing data:

# In[7]:


dataset.dropna(axis=0, inplace=True)
dataset.shape


# ### Re-Checking for Missing Data:

# In[8]:


dataset.isnull().sum()


# ## Yearly Count of Terrorist Attack-

# In[9]:


plt.figure(figsize=(15,10))
sns.countplot(x="Year", data=dataset)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# ## Observation
# From the graph we can see 2013-17 marks the highest attacks with 2014 having the highest.
# 
# There has been a gradual increase in Terror Activities since 2004.
# 
# Highest number of terror activities occurred in the year 2014.
# 
# After 2014 the terror activities started to decrease.
# 
# 

# ## Month Analysis

# In[10]:


dataset['Month'].value_counts()
plt.figure(figsize = (15, 10))
sns.countplot(x='Month', data = dataset)
plt.xticks(rotation=90)
plt.show()


# ## Terrorist Activities By Region In Each Year

# In[11]:


pd.crosstab(dataset.Year, dataset.Region).plot(kind='area',figsize=(20,10))
plt.title('Terrorist Activities By Region In Each Year')
plt.ylabel('Number of Attacks')
plt.show()


# ## Region Affected by Terrorist Attack

# In[12]:


dataset['Region'].value_counts()
plt.figure(figsize=(15,8
                   ))
sns.countplot(x='Region',data=dataset)
plt.xticks(rotation=70)
plt.show()


# ## Counting the Yearly Casualities-

# 
# year_cas = dataset.groupby('Year').Casualities.sum().to_frame().reset_index()
# year_cas.columns = ['Year','Casualities']
# px.bar(data_frame=year_cas,x = 'Year',y = 'Casualities',color='Casualities',template='plotly_dark')

# ## Observation
# It is observed that 2015 marks the highest Casualities records.
# 
# 

# ## Type of Target Attacks

# In[14]:


target = list(dataset['Target_type'])
target_map = dict(Counter(target))
target_df = pd.DataFrame(target_map.items())
target_df.columns = ['Target Type','Count']
px.bar(data_frame=target_df,x = 'Target Type',y = 'Count',color='Target Type',template='plotly_white')


# ## Observation
# Private Citizens and Property Counts the highest amongst all.

# ## Analysing the Type of Attacks:-

# In[15]:


#Counting the Casuallities according the Attack Type
AttackType=dataset.pivot_table(columns='AttackType',values='Casualities',aggfunc='sum')
AttackType = AttackType.T
AttackType['Type'] = AttackType.index


# In[16]:


#plotting the Attack Type
labels = AttackType.columns.tolist()
attack=AttackType.T
values=attack.values.tolist()
values = sum(values,[])
attack_type = list(dataset['AttackType'].unique())
fig = go.Figure(data=[go.Pie(labels = attack_type,values=values,hole=.3)])
fig.update_layout(template = 'gridon')
fig.show()


# ## Observation
# Bombing and Explosion method shows the highest chossen type.
# 
# 

# ## Count of Weapon Chssen for Attack.

# In[17]:


from collections import Counter


# In[18]:


values = list(dataset['AttackType'])
value_map = dict(Counter(values))
value_dataset = pd.DataFrame(value_map.items())
value_dataset.columns = ["AttackType","Count of Attack Type"]


# In[19]:


px.bar(data_frame=value_dataset,x = 'AttackType',y = 'Count of Attack Type',color = 'AttackType',template="plotly_white")


# ## Observation
# Again, Bombing and Explosion shows the highest.

# ## Plotting the HOT-ZONE of Terrorism on the highest year of Terrorist Attack i.e. 2014.##

# In[20]:


import folium
from folium.plugins import MarkerCluster
year=dataset[dataset['Year']==2014]
mapData=year.loc[:,'City':'longitude']
mapData=mapData.dropna().values.tolist()

map = folium.Map(location = [0, 50], tiles='CartoDB positron', zoom_start=2) 
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for point in range(0, len(mapData)):
    folium.Marker(location=[mapData[point][1],mapData[point][2]],
                  popup = mapData[point][0]).add_to(markerCluster)
map


# ## Observation
# Iraq shows the highest Terror Attacks followed by other Middle-east region.
# 
# 

# ## Top 15 Countries showing the Highest Terror Attack.

# In[21]:


plt.figure(figsize=(15,6))
country_attack=dataset.Country.value_counts()[:15].reset_index()
country_attack.columns= ["Country", "Total Attacks"]
px.bar(data_frame= country_attack,x = 'Country',y = 'Total Attacks',color = 'Country',template='plotly_white')


# ## Observation
# Iraq, again the highest followed by Pakistan, Afganistan and India.
# 
# 

# ## Counting the Total Number of Casualities in each Country.

# In[22]:


plt.figure(figsize=(15, 8))
cas_count= dataset.groupby("Country").Casualities.sum().to_frame().reset_index().sort_values("Casualities", ascending=False)[:15]
px.bar(data_frame=cas_count,x = 'Country',y = 'Casualities',color='Country',template='plotly_white')


# ## Count of Terror Attack Region - Wise.

# In[23]:


region_attacks = dataset.Region.value_counts().to_frame().reset_index()
region_attacks.columns = ['Region', 'Total Attacks']
fig = px.bar_polar(data_frame=region_attacks,r = 'Total Attacks',theta='Region',color = 'Region',
                  template="ggplot2", color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()


# ## Observation
# Middle East and North Africa shows the highest followed by South Asia.
# 
# 

# ## Conclusion
# - Hot zones of terrorism is Middle east and North Arica so,we should focus in these region.
# - Iraq, Afganistan and Pakistan most suffered country, Government should be aware from the citizens of these countries.
# - Terrorist like to target Private citizens, Army and Police mostly, Security should be tighten in all these areas.
# - BOMBING and EXPLOSIVE are most used weapon and attack type b=used by terrorist
# - Government should tighten borders and should strict arms law.
# - Most number of attacks were done by unknown group or not an group terrorist
# - All country should have to make pact to to tackle terrorism because after 2005 there is rapid increase in Terrorist Activites.¶

# # THANK YOU !
