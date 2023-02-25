#!/usr/bin/env python
# coding: utf-8

# #                                         Lets Grow More 
# 
# ##                      Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                               Name - Samiksha Makhija
# 
# # 
# 
# ## Task 2 - Stock Market Prediction And Forecasting Using Stacked LSTM
# 
# ### Datasetlinks: : https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv

# ## 
# ## Importing necessLibraries

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Checking my tensorflow version

# In[2]:


tf.__version__


# ## Loading the Dataset

# In[3]:


#Import the data and remove rows containing NAN values
df = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
df=df. iloc[::-1]
df.head()


# In[4]:


df.tail()


# ## Data Preprocessing

# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df_high=df.reset_index()['High']


# In[8]:


plt.plot(df_high)


# Since LSTM are sensitive to the scale of the data, so we apply MinMax Scaler to transform our values between 0 and 1

# In[9]:


scaler = MinMaxScaler(feature_range = (0,1))
df_high = scaler.fit_transform(np.array(df_high).reshape(-1,1))


# In[10]:


df_high.shape


# In[11]:


df_high


# ## Split the data into train and test split

# In[12]:


training_size = int(len(df_high) * 0.75)
test_size = len(df_high) - training_size
train_data, test_data = df_high[0:training_size,:], df_high[training_size:len(df_high),:1]


# In[13]:


training_size,test_size


# In[14]:


def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[15]:


time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# In[16]:


#Reshape the input to be [samples, time steps, features] which is the requirement of LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# In[17]:


print(x_train.shape), print(y_train.shape)


# In[18]:


print(x_test.shape), print(y_test.shape)


# ## Creating the LSTM Model

# In[19]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[20]:


model.summary()


# In[21]:


model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size = 64, verbose = 1)


# ## Predicting and checking performance metrics

# In[22]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# ## Transforming back to original form

# In[23]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# ## Calculating RMSE performance metrics

# In[24]:


math.sqrt(mean_squared_error(y_train, train_predict))


# ## Testing Data RMSE

# In[25]:


math.sqrt(mean_squared_error(y_test, test_predict))


# ## Plotting

# In[26]:


#Shift train prediction for plotting
look_back = 100
trainPredictPlot = np.empty_like(df_high)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#Shift test prediction for plotting
testPredictPlot = np.empty_like(df_high)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2)+1:len(df_high) - 1, :] = test_predict

#Plot baseline and predictions
plt.plot(scaler.inverse_transform(df_high))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# - Green indicates the Predicted Data
# - Blue indicates the Complete Data
# - Orange indicates the Train Data

# ## Predicting the next 28 days Stock Price

# In[27]:


len(test_data), x_test.shape


# In[28]:


x_input = test_data[409:].reshape(1,-1)
x_input.shape


# In[29]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()


# In[30]:


lst_output=[]
n_steps=100
nextNumberOfDays = 28
i=0

while(i<nextNumberOfDays):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[31]:


day_new = np.arange(1,101)
day_pred = np.arange(101,129)


# In[32]:


day_new.shape


# In[33]:


day_pred.shape


# In[34]:


df3 = df_high.tolist()
df3.extend(lst_output)


# In[35]:


len(df_high)


# In[36]:


plt.plot(day_new, scaler.inverse_transform(df_high[1935:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))


# In[37]:


df3=df_high.tolist()
df3.extend(lst_output)
plt.plot(df3[2000:])


# In[38]:


df3=scaler.inverse_transform(df3).tolist()


# In[39]:


plt.plot(df3)


# # THANK YOU ! 
