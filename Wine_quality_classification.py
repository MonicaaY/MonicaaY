#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('C:\\Users\9588504\Desktop\wine.csv')


# In[6]:


df.head(5)


# In[9]:


fig, ax = plt.subplots(figsize = (100,100))
import seaborn as sns
sns.heatmap(df, annot = True)


# In[15]:


sns.pairplot(df)


# In[13]:


sns.countplot(x = 'Wine', data = df)


# In[14]:


sns.pairplot(df, hue = 'Wine')


# In[16]:


df.columns


# In[17]:


selected_features = ['Alcohol', 'Malic.acid', 'Ash', 'Acl', 'Mg', 'Phenols',
       'Flavanoids', 'Nonflavanoid.phenols', 'Proanth', 'Color.int', 'Hue',
       'OD', 'Proline']


# In[18]:


X = df[selected_features]
y = df['Wine']


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)


# In[26]:


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train,y_train)


# In[27]:


model1.score(X_test,y_test)


# In[29]:


from sklearn.naive_bayes import GaussianNB
model2 = GaussianNB()
model2.fit(X_train,y_train)
model2.score(X_test,y_test)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier()
model3.fit(X_train,y_train)
model3.score(X_test,y_test)


# In[35]:


from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier()
model4.fit(X_train,y_train)
model4.score(X_test,y_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
model5 = LogisticRegression()
model5.fit(X_train,y_train)
model5.score(X_test,y_test)


# In[ ]:
