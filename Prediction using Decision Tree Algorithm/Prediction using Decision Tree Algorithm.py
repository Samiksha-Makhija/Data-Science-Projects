#!/usr/bin/env python
# coding: utf-8

# #                                             Lets Grow More
# 
# ##                     Virtual Internship Program - *Data Science* (Feb 2023)
# 
# #                                  Name - Samiksha Makhija
# 
# # 
# 
# ## Task 6 -Prediction using Decision Tree  Algorithm  
# 
# ### Task Description - create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to  predict the right class accordingly.  
# 
# ### Dataset : https://bit.ly/3kXTdox
# 
# 

# ## 
# ## Importing required libraries

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# ## Loading the Dataset

# In[4]:


# Reading the Dataset
data=pd.read_csv("Iris.csv")
data.head()


# ## Data Preprocessing

# ### Shape of Dataset

# In[5]:


data.shape


# ### Dataset Columns

# In[6]:


data.columns


# ### Dataset Summary

# In[7]:


data.info()


# ### Dataset Statistical Summary

# In[17]:


data.describe()


# ### Checking Null Values

# In[18]:


data.isnull().sum()


# ### Checking columns count of " Species "

# In[19]:


data['Species'].value_counts()


# ### Pie Plot to Show the overall types of Iris Classifications

# In[20]:


data['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# ## Correlation Heatmap

# In[21]:


plt.figure(figsize=(9,7))
sns.heatmap(data.corr(),cmap='CMRmap',annot=True,linewidths=2)
plt.title("Correlation Graph",size=20)
plt.show()


# ## Defining Independent and Dependent Variables

# In[22]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species


# ## Splitting the Dataset into Training and Test Sets

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)


# ## Defining the Decision Tree Classifier and Fitting the Training Set

# In[24]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# ## Visualizing the Decision Tree

# In[14]:


from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# ## Prediction on Test Data

# In[15]:


y_pred = dtree.predict(X_test)
y_pred


# ## Checking the Accuracy of the Model

# In[16]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[18]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# ## Predicting the output class for random values for petal and sepal length and width
# 
# #### Predicting the flower type for a flower with sepal length, sepal width, petal length, petal width as 5cm, 3.6cm, 1.4cm and 0.2cm respectively

# In[19]:


dtree.predict([[5, 3.6, 1.4 , 0.2]])


# #### Predict the flower type for a flower with sepal length, sepal width, petal length, petal width as 9cm, 3.1cm, 5cm and 1.5cm respectively
# 
# 

# In[20]:


dtree.predict([[9, 3.1, 5, 1.5]])


# Predict the flower type for a flower with sepal length, sepal width, petal length, petal width as 4.1cm, 3cm, 5.1cm and 1.8cm respectively

# In[21]:


dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# # THANK YOU !
