#!/usr/bin/env python
# coding: utf-8

# # Generate out-of-sample predictions with LightGBM and CatBoost

# ## Imports & Settings

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from time import time
import sys, os
from pathlib import Path

import pandas as pd
from scipy.stats import spearman

import lightgbm as lgb
from catboost import Pool

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import MultipleTimeSeriesCV


# In[3]:


sns.set_style('whitegrid')


# In[4]:


YEAR = 252
idx = pd.IndexSlice


# In[5]:


scope_params = ['lookahead', 'train_length', 'test_length']
daily_ic_metrics = ['daily_ic_mean', 'daily_ic_mean_n', 'daily_ic_median', 'daily_ic_median_n']
lgb_train_params = ['learning_rate', 'num_leaves', 'feature_fraction', 'min_data_in_leaf']
catboost_train_params = ['max_depth', 'min_child_samples']


# ## Generate LightGBM predictions

# ### Model Configuration

# In[16]:


base_params = dict(boosting='gbdt',
                   objective='regression',
                   verbose=-1)

categoricals = ['year', 'month', 'sector', 'weekday']


# In[7]:


lookahead = 1
store = Path('data/predictions.h5')


# ### Get Data

# In[18]:


data = pd.read_hdf('data/data.h5', 'model_data').sort_index()


# In[ ]:


labels = sorted(data.filter(like='_fwd').columns)
features = data.columns.difference(labels).tolist()
label = f'r{lookahead:02}_fwd'


# In[ ]:


data = data.loc[idx[:, '2010':], features + [label]].dropna()


# In[ ]:


for feature in categoricals:
    data[feature] = pd.factorize(data[feature], sort=True)[0]


# In[19]:


lgb_data = lgb.Dataset(data=data[features],
                       label=data[label],
                       categorical_feature=categoricals,
                       free_raw_data=False)


# ### Generate predictions

# In[ ]:


lgb_ic = pd.read_hdf('data/model_tuning.h5', 'lgb/ic')
lgb_ic_avg = pd.read_hdf('data/model_tuning.h5', 'lgb/ic_avg')


# In[ ]:


def get_lgb_params(data, t=5, best=0):
    param_cols = scope_params[1:] + lgb_train_params + ['boost_rounds']
    df = data[data.lookahead==t].sort_values('ic', ascending=False).iloc[best]
    return df.loc[param_cols]


# In[ ]:


for position in range(10):
    params = get_lgb_params(daily_ic_avg,
                    t=lookahead,
                    best=position)
    
    params = params.to_dict()
    
    for p in ['min_data_in_leaf', 'num_leaves']:
        params[p] = int(params[p])
    train_length = int(params.pop('train_length'))
    test_length = int(params.pop('test_length'))
    num_boost_round = int(params.pop('boost_rounds'))
    params.update(base_params)

    print(f'\nPosition: {position:02}')
    
    # 1-year out-of-sample period
    n_splits = int(YEAR / test_length)
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              test_period_length=test_length,
                              lookahead=lookahead,
                              train_period_length=train_length)

    predictions = []
    start = time()
    for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
        print(i, end=' ', flush=True)
        lgb_train = lgb_data.subset(train_idx.tolist()).construct()

        model = lgb.train(params=params,
                          train_set=lgb_train,
                          num_boost_round=num_boost_round,
                          verbose_eval=False)

        test_set = data.iloc[test_idx, :]
        y_test = test_set.loc[:, label].to_frame('y_test')
        y_pred = model.predict(test_set.loc[:, model.feature_name()])
        predictions.append(y_test.assign(prediction=y_pred))

    if position == 0:
        test_predictions = (pd.concat(predictions)
                            .rename(columns={'prediction': position}))
    else:
        test_predictions[position] = pd.concat(predictions).prediction

by_day = test_predictions.groupby(level='date')
for position in range(10):
    if position == 0:
        ic_by_day = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0]).to_frame()
    else:
        ic_by_day[position] = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0])
print(ic_by_day.describe())
test_predictions.to_hdf(store, f'lgb/test/{lookahead:02}')


# ## Generate CatBoost predictions

# ### Model Configuration

# In[7]:


lookahead = 1
store = Path('data/predictions.h5')


# ### Get Data

# In[18]:


data = pd.read_hdf('data/data.h5', 'model_data').sort_index()


# In[ ]:


labels = sorted(data.filter(like='_fwd').columns)
features = data.columns.difference(labels).tolist()
label = f'r{lookahead:02}_fwd'


# In[ ]:


data = data.loc[idx[:, '2010':], features + [label]].dropna()


# In[ ]:


for feature in categoricals:
    data[feature] = pd.factorize(data[feature], sort=True)[0]


# In[19]:


catboost_data = Pool(label=outcome_data[label],
                     data=outcome_data.drop(label, axis=1),
                     cat_features=cat_cols_idx)


# ### Generate predictions

# In[ ]:


catboost_ic = pd.read_hdf('data/model_tuning.h5', 'catboost/ic')
catboost_ic_avg = pd.read_hdf('data/model_tuning.h5', 'catboost_ic/ic_avg')


# In[ ]:


def get_cb_params(data, t=5, best=0):
    param_cols = scope_params[1:] + catboost_train_params + ['boost_rounds']
    df = data[data.lookahead==t].sort_values('ic', ascending=False).iloc[best]
    return df.loc[param_cols]


# In[ ]:


for position in range(10):
    params = get_cb_params(catboost_ic_avg,
                    t=lookahead,
                    best=position)
    
    params = params.to_dict()
    
    for p in ['max_deptn', 'min_child_samples']:
        params[p] = int(params[p])
    train_length = int(params.pop('train_length'))
    test_length = int(params.pop('test_length'))
    num_boost_round = int(params.pop('boost_rounds'))
    params['task_type'] = 'GPU'

    print(f'\nPosition: {position:02}')
    
    # 1-year out-of-sample period
    n_splits = int(YEAR / test_length)
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              test_period_length=test_length,
                              lookahead=lookahead,
                              train_period_length=train_length)

    predictions = []
    start = time()
    for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
        print(i, end=' ', flush=True)
        train_set = catboost_data.slice(train_idx.tolist())

        model = CatBoostRegressor(**params)
        model.fit(X=train_set,
                  verbose_eval=False)

        test_set = data.iloc[test_idx, :]
        y_test = test_set.loc[:, label].to_frame('y_test')
        y_pred = model.predict(test_set.loc[:, model.feature_name()])
        predictions.append(y_test.assign(prediction=y_pred))

    if position == 0:
        test_predictions = (pd.concat(predictions)
                            .rename(columns={'prediction': position}))
    else:
        test_predictions[position] = pd.concat(predictions).prediction

by_day = test_predictions.groupby(level='date')
for position in range(10):
    if position == 0:
        ic_by_day = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0]).to_frame()
    else:
        ic_by_day[position] = by_day.apply(lambda x: spearmanr(x.y_test, x[position])[0])
print(ic_by_day.describe())
test_predictions.to_hdf(store, f'catboost/test/{lookahead:02}')

