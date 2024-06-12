#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df = pd.read_csv("advertising.csv")
df.head()


# In[16]:


df.shape


# In[17]:


df.describe()


# In[18]:


sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# In[19]:


df['TV'].plot.hist(bins=10, color="red", xlabel="TV")


# In[20]:


df['Radio'].plot.hist(bins=10, color="orange", xlabel="Radio")


# In[22]:


df['Newspaper'].plot.hist(bins=10,color="green", xlabel="newspaper")


# In[23]:


sns.heatmap(df.corr(),annot = True)
plt.show()


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)


# In[25]:


print(X_train)


# In[26]:


print(X_test)


# In[28]:


print(y_train)


# In[29]:


print(y_test)


# In[30]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
res= model.predict(X_test)
print(res)


# In[31]:


print("Accuracy Score: ", model.score(X_test,y_test)*100)


# In[34]:


model.coef_


# In[35]:


model.intercept_


# In[36]:


0.05473199* 69.2 + 7.14382225


# In[37]:


plt.style.use('dark_background')
plt.grid()
plt.plot(res)


# In[38]:


plt.style.use('default')
plt.grid()
plt.scatter(X_test, y_test)
plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'r')
plt.show()


# In[ ]:




