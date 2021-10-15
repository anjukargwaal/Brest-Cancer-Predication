#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt


# In[87]:


df=pd.read_csv("data1.csv")
df.head()


# In[88]:


df.drop("id",axis=1)


# In[89]:


df=df.drop('id',axis=1)


# In[90]:


df


# In[91]:



x=df.loc[:,df.columns!='diagnosis']


# In[92]:



y=df['diagnosis']


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=55)


# In[95]:


x_test


# In[96]:


x_train


# In[97]:



from sklearn.linear_model import LogisticRegression


# In[98]:


model=LogisticRegression()


# In[99]:


model.fit(x_train,y_train)


# In[100]:


model.score(x_test,y_test)


# In[101]:


model.predict(x_test)


# In[ ]:





# In[ ]:





# In[ ]:




