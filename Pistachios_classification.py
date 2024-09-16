#!/usr/bin/env python
# coding: utf-8

# In[168]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[169]:


pista = pd.read_csv("C:\\Users\Suyash Pandey\OneDrive\Desktop\pistachio.csv")


# In[170]:


pista.head(5)


# In[171]:


pista.info()


# In[172]:


sns.countplot(x = 'Class', data = pista)


# In[173]:


pista.columns


# In[174]:


features = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
       'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'Class']


# In[175]:


#sns.pairplot(pista[features], hue = 'Class')


# In[176]:


X = pista[features].drop(labels = {'Class'}, axis = 1)
y = pista['Class']


# In[177]:


Scaler = MinMaxScaler()
X = Scaler.fit_transform(X)


# In[178]:


print(X)


# In[179]:


y


# In[180]:


pista['Class'].replace({'Kirmizi_Pistachio' : 1, 'Siit_Pistachio' : 0}, inplace = True)


# In[181]:


print(X.shape)
print(y.shape)


# In[182]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33)


# In[183]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[184]:


model1 = LogisticRegression()
model1.fit(X_train,y_train)
model1.score(X_test,y_test)


# In[185]:


model2 = KNeighborsClassifier()
model2.fit(X_train,y_train)
model2.score(X_test,y_test)


# In[186]:


model3 = DecisionTreeClassifier()
model3.fit(X_train,y_train)
model3.score(X_test,y_test)


# In[187]:


model4 = RandomForestClassifier()
model4.fit(X_train,y_train)
model4.score(X_test,y_test)


# In[188]:


y_train_predict = model1.predict(X_train)


# In[189]:


y_train_predict = (y_train_predict > 0.5)


# In[190]:


y_train_predict


# In[191]:


y_predict = model1.predict(X_test)


# In[192]:


y_predict = (y_predict > 0.5)


# In[193]:


y_predict


# In[194]:


from sklearn.metrics import confusion_matrix, classification_report
cm1 = confusion_matrix(y_train, y_train_predict)
cm2 = confusion_matrix(y_test,y_predict)
sns.heatmap(cm1, annot = True)


# In[195]:


sns.heatmap(cm2, annot = True)


# In[196]:


print('Training_Report:\n', classification_report(y_train, y_train_predict))
print('Testing_Report:\n', classification_report(y_test, y_predict))
