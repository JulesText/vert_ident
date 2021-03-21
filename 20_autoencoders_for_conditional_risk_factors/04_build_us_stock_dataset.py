#!/usr/bin/env python
# coding: utf-8

# # Download historical equity data for NASDAQ stocks from yahoo finance

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import pandas as pd

from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import yfinance as yf


# In[3]:


idx = pd.IndexSlice


# In[ ]:


results_path = Path('results', 'asset_pricing')
if not results_path.exists():
    results_path.mkdir(parents=True)


# In[4]:


def chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


# ## Get NASDAQ symbols

# In[5]:


traded_symbols = get_nasdaq_symbols()


# In[6]:


traded_symbols.info()


# ## Download metadata from yahoo finance

# ### NASDAQ symbols

# In[21]:


all_tickers = traded_symbols[~traded_symbols.ETF].index.unique().to_list()


# In[7]:


yf_tickers = yf.Tickers(all_tickers)


# Currently, there's a `yfinance` [bug](https://github.com/ranaroussi/yfinance/issues/208) that causes some stock info downloads to fail; apply the workaround described in the comments or wait for a new release to get the full dataset. Currently, we are losing a few hundred.

# In[8]:


info = []
for ticker in yf_tickers.tickers:
    try:
        info.append(pd.Series(ticker.info).to_frame(ticker.ticker))
    except Exception as e:
        print(e, ticker.ticker)
info = pd.concat(info, axis=1).dropna(how='all').T
info = info.apply(pd.to_numeric, errors='ignore')
info.to_hdf(results_path / 'data.h5', 'stocks/info')


# ## Download adjusted price data using yfinance

# In[22]:


prices_adj = []
with pd.HDFStore('chunks.h5') as store:
    for i, chunk in enumerate(chunks(all_tickers, 100)):
        print(i, end=' ', flush=True)
        prices_adj.append(yf.download(chunk, period='max', auto_adjust=True).stack(-1))


# In[23]:


prices_adj = (pd.concat(prices_adj)
              .dropna(how='all', axis=1)
              .rename(columns=str.lower)
              .swaplevel())


# In[24]:


prices_adj.index.names = ['ticker', 'date']


# In[25]:


len(prices_adj.index.unique('ticker'))


# ### Remove outliers

# In[26]:


df = prices_adj.close.unstack('ticker')
pmax = df.pct_change().max()
pmin = df.pct_change().min()
to_drop = pmax[pmax > 1].index.union(pmin[pmin<-1].index)
len(to_drop)


# In[27]:


prices_adj = prices_adj.drop(to_drop, level='ticker')


# In[28]:


len(prices_adj.index.unique('ticker'))


# In[29]:


prices_adj.sort_index().loc[idx[:, '1990': '2019'], :].to_hdf(results_path / 'data.h5', 
                                                              'stocks/prices/adjusted')

