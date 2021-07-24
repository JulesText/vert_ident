#!/usr/bin/env python 
# coding: utf-8

# # Preparing Alpha Factors and Features to predict Stock Returns

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from talib import RSI, BBANDS, MACD, ATR


# In[3]:


MONTH = 21
YEAR = 12 * MONTH


# In[4]:


START = '2013-01-01'
END = '2017-12-31'


# In[5]:


sns.set_style('whitegrid')
idx = pd.IndexSlice


# ## Loading Quandl Wiki Stock Prices & Meta Data

# In[6]:


DATA_STORE = '../data/assets.h5'
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .swaplevel()
              .sort_index())
    prices.volume /= 1e3
    stocks = (store['us_equities/stocks']
              .loc[:, ['marketcap', 'ipoyear', 'sector']])


# ## Remove stocks with few observations

# In[7]:


min_obs = 2 * YEAR
nobs = prices.groupby(level='ticker').size()
keep = nobs[nobs > min_obs].index
prices = prices.loc[idx[keep, :], :]


# ### Align price and meta data

# In[8]:


stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
stocks.sector = stocks.sector.str.lower().str.replace(' ', '_')
stocks.index.name = 'ticker'


# In[9]:


shared = (prices.index.get_level_values('ticker').unique()
          .intersection(stocks.index))
stocks = stocks.loc[shared, :]
prices = prices.loc[idx[shared, :], :]


# In[10]:


prices.info(null_counts=True)


# In[11]:


stocks.info()


# In[12]:


stocks.sector.value_counts()


# ## Compute Rolling Average Dollar Volume

# In[13]:


# compute dollar volume to determine universe
prices['dollar_vol'] = prices.loc[:, 'close'].mul(prices.loc[:, 'volume'], axis=0)
prices['dollar_vol'] = (prices
                        .groupby('ticker',
                                 group_keys=False,
                                 as_index=False)
                        .dollar_vol
                        .rolling(window=21)
                        .mean()
                        .fillna(0)
                        .reset_index(level=0, drop=True))
prices.dollar_vol /= 1e3


# In[14]:


prices['dollar_vol_rank'] = (prices
                             .groupby('date')
                             .dollar_vol
                             .rank(ascending=False))


# ## Add some Basic Factors

# ### Compute the Relative Strength Index

# In[15]:


prices['rsi'] = prices.groupby(level='ticker').close.apply(RSI)


# In[16]:


ax = sns.distplot(prices.rsi.dropna())
ax.axvline(30, ls='--', lw=1, c='k')
ax.axvline(70, ls='--', lw=1, c='k')
ax.set_title('RSI Distribution with Signal Threshold')
plt.tight_layout();


# ### Compute Bollinger Bands

# In[17]:


def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)


# In[18]:


prices = (prices.join(prices
                      .groupby(level='ticker')
                      .close
                      .apply(compute_bb)))


# In[19]:


prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)


# In[20]:


fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_low'].dropna(), ax=axes[0])
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_high'].dropna(), ax=axes[1])
plt.tight_layout();


# ### Compute Average True Range

# In[21]:


def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low,
             stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())


# In[22]:


prices['atr'] = (prices.groupby('ticker', group_keys=False)
                 .apply(compute_atr))


# In[23]:


sns.distplot(prices[prices.dollar_vol_rank<50].atr.dropna());


# ### Compute Moving Average Convergence/Divergence

# In[24]:


def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)


# In[25]:


prices['macd'] = (prices
                  .groupby('ticker', group_keys=False)
                  .close
                  .apply(compute_macd))


# In[26]:


prices.macd.describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999]).apply(lambda x: f'{x:,.1f}')


# In[27]:


sns.distplot(prices[prices.dollar_vol_rank<100].macd.dropna());


# ## Compute Lagged Returns

# In[28]:


lags = [1, 5, 10, 21, 42, 63]


# In[29]:


returns = prices.groupby(level='ticker').close.pct_change()
percentiles=[.0001, .001, .01]
percentiles+= [1-p for p in percentiles]
returns.describe(percentiles=percentiles).iloc[2:].to_frame('percentiles').style.format(lambda x: f'{x:,.2%}')


# In[30]:


q = 0.0001


# ### Winsorize outliers

# In[31]:


for lag in lags:
    prices[f'return_{lag}d'] = (prices.groupby(level='ticker').close
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(q),
                                                       upper=x.quantile(1 - q)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1)
                                )


# ### Shift lagged returns

# In[32]:


for t in [1, 2, 3, 4, 5]:
    for lag in [1, 5, 10, 21]:
        prices[f'return_{lag}d_lag{t}'] = (prices.groupby(level='ticker')
                                           [f'return_{lag}d'].shift(t * lag))


# ## Compute Forward Returns

# In[33]:


for t in [1, 5, 10, 21]:
    prices[f'target_{t}d'] = prices.groupby(level='ticker')[f'return_{t}d'].shift(-t)


# ## Combine Price and Meta Data

# In[34]:


prices = prices.join(stocks[['sector']])


# ## Create time and sector dummy variables

# In[35]:


prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month


# In[36]:


prices.info(null_counts=True)


# In[37]:


prices.assign(sector=pd.factorize(prices.sector, sort=True)[0]).to_hdf('data.h5', 'model_data/no_dummies')


# In[38]:


prices = pd.get_dummies(prices,
                        columns=['year', 'month', 'sector'],
                        prefix=['year', 'month', ''],
                        prefix_sep=['_', '_', ''],
                        drop_first=True)


# In[39]:


prices.info(null_counts=True)


# ## Store Model Data

# In[40]:


prices.to_hdf('data.h5', 'model_data')


# ## Explore Data

# ### Plot Factors

# In[41]:


target = 'target_5d'
top100 = prices[prices.dollar_vol_rank<100].copy()


# ### RSI

# In[42]:


top100.loc[:, 'rsi_signal'] = pd.cut(top100.rsi, bins=[0, 30, 70, 100])


# In[43]:


top100.groupby('rsi_signal')['target_5d'].describe()


# ### Bollinger Bands

# In[44]:


j=sns.jointplot(x=top100.bb_low, y=target, data=top100)
j.annotate(pearsonr);


# In[45]:


j=sns.jointplot(x='bb_high', y=target, data=top100)
j.annotate(pearsonr);


# ### ATR

# In[46]:


j=sns.jointplot(x='atr', y=target, data=top100)
j.annotate(pearsonr);


# ### MACD

# In[47]:


j=sns.jointplot(x='macd', y=target, data=top100)
j.annotate(pearsonr);
