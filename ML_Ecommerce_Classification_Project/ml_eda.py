#!/usr/bin/env python
# coding: utf-8

# # ML project (Part 01 : EDA)
# - Objective : Predict if user will purchase a given product in the cart or not.
# - Dataset : 
#   - This dataset contains behavior data for 7 months (from October 2019 to April 2020) from a large multi-category online store.
#   - Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users.
#   - This dataset is collected by Open CDP project.
#   - https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv

# # Set up 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import cm
from datetime import datetime
import seaborn as sns


# # Read in data

# In[2]:


# Here is the sampled I drew from the original dataset : 1 million data from each month, 7 millions data in total.
df = pd.read_csv("/Users/crystal/Desktop/final projects/test/df_sample.csv")


# In[5]:


# Size of dataset
df.shape


# In[6]:


df.head()


# In[7]:


# Check the basic informations of the dataset
df.info()


# In[8]:


# Variables in the dataset
df.columns


# # Exploratory Data Analysis

# ## 1. Overview

# **Number of Customers**

# In[9]:


visitor = df['user_id'].nunique()
print ("visitors: {}".format(visitor))


# **Number of Products**

# In[10]:


product = df['product_id'].nunique()
print ("products: {}".format(product))


# **Number of Products under Each Brand**

# In[11]:


brands = df['brand'].value_counts()


# In[12]:


# Plot distribution of the top 30 brands
top_brands = brands.head(30)
plt.figure(figsize=(10,6))
sns.barplot(x=top_brands.values, y=top_brands.index, palette='viridis')
plt.title('Top 30 Brands', fontsize=16)
plt.xlabel('Number of products', fontsize=14)
plt.ylabel('Brands', fontsize=14)
plt.show()


# Three types of events : view, cart & purchase

# In[13]:


events = df['event_type'].value_counts()
events


# In[14]:


labels = ['view', 'cart','purchase']
size = df['event_type'].value_counts()
colors = ['yellowgreen', 'lightskyblue','lightcoral']
explode = [0, 0.1,0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Event_Type', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# ## 2. Deeper look into products, users & events

# **Create three tables to store the unique informations of products, users, events.**
# 
# - products: Holds unique information from the products
# - user_sessions table: Holds unique information from the website sessions
# - events: Holds unique information from the website events

# In[3]:


# Creating the products table
products = df[['product_id','category_id','category_code','brand','price']].drop_duplicates()
products.set_index('product_id',inplace=True)

# Creating the user_sessions table
user_sessions = df[['user_session','user_id']].drop_duplicates()
user_sessions.set_index('user_session',inplace=True)

# creating the events table
events = df[['event_time','event_type','product_id','user_session']]


# ### 1). Products

# **What are the top 10 products by in terms of its earnings?**

# In[4]:


product_earnings = df[df['event_type'] == 'purchase'].groupby('product_id')['price'].sum()
top_10_products = product_earnings.sort_values(ascending=False).head(10)
print('Top 10 products by earnings:')
print(top_10_products)


# **Comment:** <br>
# 1005115 and 1005105 are two products that has taken around 40% of earnings among the top 10 products. 

# **What are the top 10 products by earnings per session ?** <br>
# Total revenue is not a meaningful metric on its own, especially if we ignore the number of sessions that landed on the product page. We need to know the conversion rate of traffic to a page. 

# In[17]:


# Calculate the earnings per session for each product
earnings_per_session = events.groupby('product_id')['event_type'].apply(lambda x: (x=='purchase').sum()/len(x.dropna().unique()))

# Merge with the products table to get the product names and categories
top_products = pd.merge(earnings_per_session, products, left_index=True, right_index=True)

# Sort by earnings per session and select the top 10 products
top_products = top_products.sort_values('event_type', ascending=False).head(10)

# Print the top 10 products by earnings per session
print(top_products[['category_code', 'brand', 'price', 'event_type']])


# **What are the top 10 products with the highest conversion rates ?**

# In[7]:


# Group the events by product ID and calculate the total sales and total views for each product
product_data = events.groupby('product_id')['event_type'].agg(['count', lambda x: sum(x=='purchase')]).rename(columns={'count': 'total_views', '<lambda_0>': 'total_sales'})

# Calculate the conversion rate for each product
product_data['conversion_rate'] = product_data['total_sales'] / product_data['total_views']

# Merge with the products table to get the product names and categories
top_products = pd.merge(product_data['conversion_rate'], products, left_index=True, right_index=True)

# Sort by conversion rate and select the top 10 products
top_products = top_products.sort_values('conversion_rate', ascending=False).head(10)

# Print the top 10 products by conversion rate
print(top_products[['category_code', 'brand', 'price', 'conversion_rate']])


# In[ ]:


# 1) Found out how many sales each product made
num_of_sales = pd.Series(events_on_products[events_on_products['event_type'] == 'purchase']['product_id'].value_counts())
num_of_sales.rename('total_sales',inplace=True)

# 2) Grouped the number of views and the number of sales by the product_id
product_data = revenue_and_views.merge(num_of_sales,right_index=True,left_index=True)

# 3) Divided the number of sales by the number of views and found out the conversion rate of all the products
cvr = pd.Series(product_data['total_sales']/product_data['total_views'])
top_10_cvr = cvr.sort_values(ascending=False).head(10)
top_10_cvr.plot(kind='bar',title='Top 10 Products by Conversion Rates',xlabel='Product_id',ylabel='Conversion Rate (from 0 to 1)')


# ### 2). Customers

# In[8]:


# 1) Merged the user_sessions table with the events table 
sessions_on_events = user_sessions.merge(events,how='left',left_on='user_session',right_on='user_session')


# **Average Order Value**

# In[9]:


# 1) Merged products table with the sessions_on_events table in order to group the products bought by each individual user_session
products_bought_by_session = sessions_on_events[sessions_on_events['event_type'] == 'purchase'][['user_session','product_id']]
session_and_price = products_bought_by_session.merge(products,how='left',left_on='product_id',right_on='product_id')[['user_session','price']]

# 2) Grouped by user_session and summed the price of goods purchased
sum_products_bought_per_session = session_and_price.groupby(by='user_session').sum()
sum_products_bought_per_session.sort_values(by='price',ascending=False)
 
# 3) Divided the price of all purchased products by the number of sessions that resulted in a purchase
aov = sum_products_bought_per_session.sum()/len(sum_products_bought_per_session.index)

print(f"The Average Order Value of each purchase is {aov}")


# **Number of purchases per customer**

# In[14]:


# 1) Left joined the user_id on the session_and_price table in order to count the unique values
purchase_sessions_and_customers = session_and_price.merge(user_sessions,how='left',left_on='user_session',right_index=True)

# 2) Divided the number of unique sessions that resulted in a purchase by the number of customers
purchase_p_customer = purchase_sessions_and_customers['user_session'].nunique()/purchase_sessions_and_customers['user_id'].nunique()

print(f"The Average number of purchases per customer is {round(purchase_p_customer,2)}.")


# **Which category customers interact the most?**

# In[11]:


top_category_n = 30
top_category = df.loc[:,'category_code'].value_counts()[:top_category_n].sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=top_category.values, y=top_category.index, palette='viridis')
plt.title(f'Top {top_category_n} categories', fontsize=16)
plt.xlabel('Number of products', fontsize=14)
plt.ylabel('Category code', fontsize=14)
plt.show()


# **Comment:** We can tell from graph that construction tools light customer interact the most and extremely more than other categories. Top 5 categories are all eletronics that customer interact the most. 

# ### 3) Events

# **Vistors Daily Trend : Does traffic flunctuate by date?**

# In[12]:


d = df.loc[:,['event_time','user_id']]
d['event_time'] = d['event_time'].apply(lambda s: str(s)[0:10])
visitor_by_date = d.drop_duplicates().groupby(['event_time'])['user_id'].agg(['count']).sort_values(by=['event_time'], ascending=True)


# In[13]:


x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
y = visitor_by_date['count']
plt.rcParams['figure.figsize'] = (20,8)

plt.plot(x,y)
plt.show()


# **Comment:**
# The plot shows that the traffic is normally around 30000 but has peaked at 70000 (becomes as twice as normal level) at 2019-10 and also drops to extremly low level like around 3000 at 2020-03. 
