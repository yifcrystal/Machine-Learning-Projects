#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[6]:


df_oct = pd.read_csv("/Users/crystal/Desktop/final projects/test/2019-Oct.csv")


# In[8]:


# Shuffle the DataFrame
df_10 = df_oct.sample(frac=1, random_state=42)
df_10 = df_10.head(1000000)
df_10.to_csv('df_10.csv', index=False)


# In[7]:


df_nov = pd.read_csv("/Users/crystal/Desktop/final projects/test/2019-Nov.csv")


# In[8]:


# Shuffle the DataFrame
df_11 = df_nov.sample(frac=1, random_state=42)
df_11 = df_11.head(1000000)
df_11.to_csv('df_11.csv', index=False)


# In[ ]:


df_dec = pd.read_csv("/Users/crystal/Desktop/final projects/test/2019-Dec.csv")


# In[ ]:


# Shuffle the DataFrame
df_12 = df_dec.sample(frac=1, random_state=42)
df_12 = df_12.head(1000000)
df_12.to_csv('df_12.csv', index=False)


# In[3]:


df_jan = pd.read_csv("/Users/crystal/Desktop/final projects/test/2020-Jan.csv")


# In[4]:


# Shuffle the DataFrame
df_01 = df_jan.sample(frac=1, random_state=42)
df_01 = df_01.head(1000000)
df_01.to_csv('df_01.csv', index=False)


# In[5]:


df_feb = pd.read_csv("/Users/crystal/Desktop/final projects/test/2020-Feb.csv")


# In[6]:


# Shuffle the DataFrame
df_02 = df_feb.sample(frac=1, random_state=42)
df_02 = df_02.head(1000000)
df_02.to_csv('df_02.csv', index=False)


# In[9]:


df_mar = pd.read_csv("/Users/crystal/Desktop/final projects/test/2020-Mar.csv")


# In[10]:


# Shuffle the DataFrame
df_03 = df_mar.sample(frac=1, random_state=42)
df_03 = df_03.head(1000000)
df_03.to_csv('df_03.csv', index=False)


# In[12]:


df_apr = pd.read_csv("/Users/crystal/Desktop/final projects/test/2020-Apr.csv")


# In[13]:


# Shuffle the DataFrame
df_04 = df_apr.sample(frac=1, random_state=42)
df_04 = df_04.head(1000000)
df_04.to_csv('df_04.csv', index=False)


# In[10]:


df_combined = pd.concat([df_10,df_11,df_12,df_01,df_02,df_03,df_04])


# In[11]:


df_combined.to_csv('df_sample.csv', index=False)


# In[12]:


df_combined.shape

