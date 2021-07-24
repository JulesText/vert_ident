#!/usr/bin/env python
# coding: utf-8

# # Performance Analysis with Alphalens

# This notebook uses the environment `ml4t-zipline`.

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from alphalens.tears import (create_returns_tear_sheet,
                             create_summary_tear_sheet,
                             create_full_tear_sheet)

from alphalens.performance import mean_return_by_quantile
from alphalens.plotting import plot_quantile_returns_bar
from alphalens.utils import get_clean_factor_and_forward_returns, rate_of_return


# In[3]:


sns.set_style('whitegrid')


# In[4]:


np.random.seed(42)
idx = pd.IndexSlice


# In[5]:


results_path = Path('results', 'asset_pricing')
if not results_path.exists():
    results_path.mkdir(parents=True)


# ## Alphalens Analysis

# ### Load predictions

# In[6]:


DATA_STORE = Path(results_path / 'data.h5')


# In[15]:


predictions = pd.read_hdf(results_path / 'predictions.h5', 'predictions_avg')


# In[16]:


factor = (predictions
          .unstack('ticker')
#           .resample('W-FRI')
#           .last()
          .stack()
          .tz_localize('UTC', level='date')
          .sort_index())
tickers = factor.index.get_level_values('ticker').unique()


# ### Get trade prices

# In[9]:


def get_trade_prices(tickers):
    prices = pd.read_hdf(DATA_STORE, 'stocks/prices/adjusted')
    prices.index.names = ['ticker', 'date']
    prices = prices.loc[idx[tickers, '2014':'2020'], 'open']
    return (prices
            .unstack('ticker')
            .sort_index()
            .shift(-1)
            .tz_localize('UTC'))


# In[10]:


trade_prices = get_trade_prices(tickers)


# In[11]:


trade_prices.info()


# ### Generate tearsheet input

# In[17]:


factor_data = get_clean_factor_and_forward_returns(factor=factor,
                                                   prices=trade_prices,
                                                   quantiles=5,
                                                   periods=(5, 10, 21)).sort_index()
factor_data.info()


# ### Create Tearsheet

# In[18]:


create_summary_tear_sheet(factor_data)


# In[19]:


create_full_tear_sheet(factor_data)


# In[ ]:




