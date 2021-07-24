#!/usr/bin/env python
# coding: utf-8

# # Train a Deep NN to predict Asset Price movements

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path

import numpy as np
import pandas as pd


# In[3]:


np.random.seed(42)


# In[4]:


idx = pd.IndexSlice


# ## Build daily dataset

# In[5]:


DATA_DIR = Path('..', 'data')


# In[6]:


prices = (pd.read_hdf(DATA_DIR / 'assets.h5', 'quandl/wiki/prices')
          .loc[idx['2010':'2017', :], ['adj_close', 'adj_volume']])
prices.info()


# ### Select most traded stocks

# In[7]:


n_dates = len(prices.index.unique('date'))
dollar_vol = (prices.adj_close.mul(prices.adj_volume)
              .unstack('ticker')
              .dropna(thresh=int(.95 * n_dates), axis=1)
              .rank(ascending=False, axis=1)
              .stack('ticker'))


# In[8]:


most_traded = dollar_vol.groupby(level='ticker').mean().nsmallest(500).index


# In[9]:


returns = (prices.loc[idx[:, most_traded], 'adj_close']
           .unstack('ticker')
           .pct_change()
           .sort_index(ascending=False))
returns.info()


# ### Stack 21-day time series

# In[10]:


n = len(returns)
T = 21 # days
tcols = list(range(T))
tickers = returns.columns


# In[11]:


data = pd.DataFrame()
for i in range(n-T-1):
    df = returns.iloc[i:i+T+1]
    date = df.index.max()
    data = pd.concat([data, 
                      df.reset_index(drop=True).T
                      .assign(date=date, ticker=tickers)
                      .set_index(['ticker', 'date'])])
data = data.rename(columns={0: 'label'}).sort_index().dropna()
data.loc[:, tcols[1:]] = (data.loc[:, tcols[1:]].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))
data.info()


# In[12]:


data.shape


# In[13]:


data.to_hdf('data.h5', 'returns_daily')


# ## Build weekly dataset

# We load the Quandl adjusted stock price data:

# In[14]:


prices = (pd.read_hdf(DATA_DIR / 'assets.h5', 'quandl/wiki/prices')
          .adj_close
          .unstack().loc['2007':])
prices.info()


# ### Resample to weekly frequency

# We start by generating weekly returns for close to 2,500 stocks without missing data for the 2008-17 period, as follows:

# In[15]:


returns = (prices
           .resample('W')
           .last()
           .pct_change()
           .loc['2008': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns.info()


# In[16]:


returns.head().append(returns.tail())


# ### Create & stack 52-week sequences

# We'll use 52-week sequences, which we'll create in a stacked format:

# In[17]:


n = len(returns)
T = 52 # weeks
tcols = list(range(T))
tickers = returns.columns


# In[18]:


data = pd.DataFrame()
for i in range(n-T-1):
    df = returns.iloc[i:i+T+1]
    date = df.index.max()    
    data = pd.concat([data, (df.reset_index(drop=True).T
                             .assign(date=date, ticker=tickers)
                             .set_index(['ticker', 'date']))])
data.info()


# In[19]:


data[tcols] = (data[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))


# In[20]:


data = data.rename(columns={0: 'fwd_returns'})


# In[21]:


data['label'] = (data['fwd_returns'] > 0).astype(int)


# In[22]:


data.shape


# In[23]:


data.sort_index().to_hdf('data.h5', 'returns_weekly')

