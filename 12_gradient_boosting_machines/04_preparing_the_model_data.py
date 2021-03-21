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
import talib
from talib import RSI, BBANDS, MACD, ATR


# In[3]:


MONTH = 21
YEAR = 12 * MONTH


# In[4]:


START = '2010-01-01'
END = '2017-12-31'


# In[5]:


sns.set_style('darkgrid')
idx = pd.IndexSlice


# In[6]:


percentiles = [.001, .01, .02, .03, .04, .05]
percentiles += [1-p for p in percentiles[::-1]]


# In[7]:


T = [1, 5, 10, 21, 42, 63]


# ## Loading Quandl Wiki Stock Prices & Meta Data

# In[8]:


DATA_STORE = '../data/assets.h5'
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .swaplevel()
              .sort_index())
    metadata = (store['us_equities/stocks'].loc[:, ['marketcap', 'sector']])


# In[9]:


prices.volume /= 1e3
prices.index.names = ['symbol', 'date']
metadata.index.name = 'symbol'


# ## Remove stocks with insufficient observations

# In[10]:


min_obs = 7 * YEAR
nobs = prices.groupby(level='symbol').size()
keep = nobs[nobs > min_obs].index
prices = prices.loc[idx[keep, :], :]


# ### Align price and meta data

# In[11]:


metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
metadata.sector = metadata.sector.str.lower().str.replace(' ', '_')


# In[12]:


shared = (prices.index.get_level_values('symbol').unique()
          .intersection(metadata.index))
metadata = metadata.loc[shared, :]
prices = prices.loc[idx[shared, :], :]


# ### Limit universe to 1,000 stocks with highest market cap

# In[13]:


universe = metadata.marketcap.nlargest(1000).index
prices = prices.loc[idx[universe, :], :]
metadata = metadata.loc[universe]


# In[14]:


metadata.sector.value_counts()


# In[15]:


prices.info(null_counts=True)


# In[16]:


metadata.info()


# ### Compute Rolling Average Dollar Volume

# In[17]:


# compute dollar volume to determine universe
prices['dollar_vol'] = prices.loc[:, 'close'].mul(prices.loc[:, 'volume'], axis=0).div(1e3)
prices['dollar_vol'] = (prices
                        .groupby('symbol',
                                 group_keys=False,
                                 as_index=False)
                        .dollar_vol
                        .rolling(window=21)
                        .mean()
                        .fillna(0)
                        .reset_index(level=0, drop=True))


# In[18]:


prices['dollar_vol_rank'] = (prices
                             .groupby('date')
                             .dollar_vol
                             .rank(ascending=False))
prices = prices.drop('dollar_vol', axis=1)


# ## Add some Basic Factors

# ### Compute the Relative Strength Index

# In[19]:


prices['rsi'] = prices.groupby(level='symbol').close.apply(RSI)


# In[20]:


ax = sns.distplot(prices.rsi.dropna())
ax.axvline(30, ls='--', lw=1, c='k')
ax.axvline(70, ls='--', lw=1, c='k')
ax.set_title('RSI Distribution with Signal Threshold')
sns.despine()
plt.tight_layout();


# ### Compute Bollinger Bands

# In[21]:


def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)


# In[22]:


prices = (prices.join(prices
                      .groupby(level='symbol')
                      .close
                      .apply(compute_bb)))


# In[23]:


prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)


# In[24]:


fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_low'].dropna(), ax=axes[0])
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_high'].dropna(), ax=axes[1])
sns.despine()
plt.tight_layout();


# ### Compute Average True Range

# In[25]:


prices['NATR'] = prices.groupby(level='symbol', 
                                group_keys=False).apply(lambda x: 
                                                        talib.NATR(x.high, x.low, x.close))


# In[26]:


def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low, 
             stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())


# In[27]:


prices['ATR'] = (prices.groupby('symbol', group_keys=False)
                 .apply(compute_atr))


# ### Compute Moving Average Convergence/Divergence

# In[28]:


prices['PPO'] = prices.groupby(level='symbol').close.apply(talib.PPO)


# In[29]:


def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)


# In[30]:


prices['MACD'] = (prices
                  .groupby('symbol', group_keys=False)
                  .close
                  .apply(compute_macd))


# ### Combine Price and Meta Data

# In[31]:


metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
prices = prices.join(metadata[['sector']])


# ## Compute Returns

# ### Historical Returns

# In[32]:


by_sym = prices.groupby(level='symbol').close
for t in T:
    prices[f'r{t:02}'] = by_sym.pct_change(t)


# ### Daily historical return quantiles

# In[33]:


for t in T:
    prices[f'r{t:02}dec'] = (prices[f'r{t:02}'].groupby(level='date')
             .apply(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')))


# ### Daily sector return quantiles

# In[34]:


for t in T:
    prices[f'r{t:02}q_sector'] = (prices.groupby(
        ['date', 'sector'])[f'r{t:02}'].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')))


# ### Compute Forward Returns

# In[35]:


for t in [1, 5, 21]:
    prices[f'r{t:02}_fwd'] = prices.groupby(level='symbol')[f'r{t:02}'].shift(-t)


# ## Remove outliers

# In[36]:


prices[[f'r{t:02}' for t in T]].describe()


# In[37]:


outliers = prices[prices.r01>1].index.get_level_values('symbol').unique()


# In[38]:


prices = prices.drop(outliers, level='symbol')


# ## Create time and sector dummy variables

# In[39]:


prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month
prices['weekday'] = prices.index.get_level_values('date').weekday


# ## Store Model Data

# In[40]:


prices.info(null_counts=True)


# In[41]:


prices.drop(['open', 'close', 'low', 'high', 'volume'], axis=1).to_hdf('data.h5', 'model_data')

