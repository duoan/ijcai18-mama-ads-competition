
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from utils import (
    load_pickle, dump_pickle, get_nominal_dfal, feats_root, mem_usage, reduce_mem_usage,
    nominal_cate_cols, ordinal_cate_cols, identity_cols, continual_cols, 
)

pd.set_option('display.max_columns', 1000)


# In[2]:


def gen_target_agg_features(data, last_da, win_das, col):
    data = data.copy()
    indexing = (dfal.da < last_da) & (dfal.da >= last_da - win_das)
    gp = data.loc[indexing, [col, 'is_trade']].groupby(col)['is_trade']
    avgs = gp.mean()
    sums = gp.sum()
    cnts = gp.size()
    indexing = data.da == last_da
    data.loc[indexing, 'agg_target_avg_{}_wd_{}'.format(col, win_das)] = data.loc[indexing, col].map(avgs)
    data.loc[indexing, 'agg_target_sum_{}_wd_{}'.format(col, win_das)] = data.loc[indexing, col].map(sums)
    data.loc[indexing, 'agg_target_cnt_{}_wd_{}'.format(col, win_das)] = data.loc[indexing, col].map(cnts)
    return data


# In[3]:


def gen_target_aggs(col, updata=False):
    feat_path = os.path.join(feats_root,'target_aggs_{}.pkl'.format(col))
    if os.path.exists(feat_path) and updata == False:
        print('Found ' + feat_path)
    else:
        print('Generating ' + feat_path)
        dfal = get_nominal_dfal()[[col, 'da', 'is_trade']]
        dmax = dfal.da.max()
        dmin = dfal.da.min()
        for da in sorted(dfal.da.unique())[1:]:
            for win_das in [1,2,3,4,5,6]:
                if da - win_das < dmin:
                    continue
                dfal = gen_target_agg_features(dfal, da, win_das, col)
        dfal = dfal.loc[dfal.da>17,:]
        dfal.drop(['is_trade'], inplace=True, axis=1)
        dfal.drop_duplicates([col, 'da'], inplace=True)
        dfal.fillna(0, inplace=True)
        dfal, _ = reduce_mem_usage(dfal)
        dump_pickle(dfal, feat_path)


# In[4]:


def gen_target_features():
    for c in tqdm_notebook(nominal_cate_cols + ordinal_cate_cols + identity_cols):
        gen_target_aggs(c)


# In[5]:


def add_target_features(data, col):
    feat_path = os.path.join(feats_root,'target_aggs_{}.pkl'.format(col))
    if not os.path.exists(feat_path):
        gen_target_aggs(col)
    agg = load_pickle(feat_path)
    return pd.merge(data, agg, how='left',on=[col, 'da'])


# In[6]:


if __name__ == '__main__':
    gen_target_features()


# # In[7]:


# dfal = get_nominal_dfal()


# # In[8]:


# dfal.shape


# # In[9]:


# dfal = dfal.loc[dfal.da>20,:]


# # In[10]:


# for c in tqdm_notebook(nominal_cate_cols + ordinal_cate_cols + identity_cols):
#     dfal = add_target_features(dfal, c)


# # In[11]:


# del dfal['dt']
# for c in dfal.columns:
#     if c.endswith('_wd_6'):
#         del dfal[c]


# # In[12]:


# dfal, _ = reduce_mem_usage(dfal)


# # In[13]:


# dfal.columns.values


# # In[14]:


# X_tr = dfal.loc[dfal.da<=22,:].drop(['da', 'hm', 'instance_id', 'is_trade'] + identity_cols, axis=1)
# y_tr = dfal.loc[dfal.da<=22,'is_trade']
# X_va = dfal.loc[dfal.da==23,:].drop(['da', 'hm', 'instance_id', 'is_trade'] + identity_cols, axis=1)
# y_va = dfal.loc[dfal.da==23,'is_trade']


# # In[15]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# import catboost as cb
# import xgboost as xg
# import lightgbm as lg


# # In[16]:


# def print_feature_importance_lgb(gbm):
#     print(80 * '*')
#     print(31 * '*' + 'Feature Importance' + 31 * '*')
#     print(80 * '.')
#     print("\n".join((".%50s => %9.5f" % x) for x in sorted(
#         zip(gbm.feature_name(), gbm.feature_importance("gain")),
#         key=lambda x: x[1],
#         reverse=True)))
#     print(80 * '.')

# def fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols):
#     params = {
#         'max_depth': 8,
#         'num_leaves': 128,
#         'objective':'binary',
#         'min_data_in_leaf': 20,
#         'learning_rate': 0.01,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'subsample':0.85,
#         'bagging_freq': 1,
#         'random_state':2018,
#         'metric': ['binary_logloss'],
#         'num_threads': 16,
#         #'is_unbalance': True
#     }

#     MAX_ROUNDS = 10000
#     dtr = lg.Dataset(X_tr, label=y_tr, categorical_feature=cates_cols)
#     dva = lg.Dataset(X_va, label=y_va, categorical_feature=cates_cols, reference=dtr)
    
#     cls = lg.train(
#         params,
#         dtr,
#         num_boost_round=MAX_ROUNDS,
#         valid_sets=(dva, dtr),
#         valid_names=['valid', 'train'],
#         early_stopping_rounds=125,
#         verbose_eval=50)
#     print_feature_importance_lgb(cls)
#     lg.plot_importance(cls, importance_type='gain', figsize=(11,12), max_num_features=50, grid=False)
#     return cls


# # In[17]:


# gbm = fit_lgb(X_tr, y_tr, X_va, y_va, nominal_cate_cols)


# # ## CatBoostClassifier

# # In[18]:


# cates_idx = [X_tr.columns.values.tolist().index(c) for c in nominal_cate_cols]


# # In[19]:


# import operator
# def verbose_feature_importance_cat(cls, X_tr):
#     cat_feature_importance = {
#         X_tr.columns.values.tolist()[idx]: score
#         for idx, score in enumerate(cls.feature_importances_)
#     }
    
#     cat_feature_importance = sorted(cat_feature_importance.items(), 
#                                     key=operator.itemgetter(1), 
#                                     reverse=False)
    
#     print(80 * '*')
#     print(31 * '*' + 'Feature Importance' + 31 * '*')
#     print(80 * '.')
#     for feature, score in reversed(cat_feature_importance):
#         print(".%50s => %9.5f" % (feature, score))
#     print(80 * '.')
    
#     feature_score = pd.DataFrame(cat_feature_importance, columns=['Feature','Score'])
    
#     plt.rcParams["figure.figsize"] = (11, 12)
#     ax = feature_score.tail(50).plot('Feature', 'Score', kind='barh', color='b')
#     ax.set_title("Catboost Feature Importance Ranking", fontsize=8)
#     ax.set_xlabel('')
#     rects = ax.patches
#     # get feature score as labels round to 2 decimal
#     labels = feature_score.tail(50)['Score'].round(2)
#     for rect, label in zip(rects, labels):
#         width = rect.get_width()
#         ax.text(width + 0.2,rect.get_y()+0.02, label, ha='center', va='bottom')
#     plt.show()


# def fit_cat(X_tr, y_tr, X_va, y_va, cates_idx):
#     print('Fitting CatBoostClassifier ...')
#     cls = cb.CatBoostClassifier(
#         iterations=2000,
#         od_type='Iter',
#         od_wait=120,
#         max_depth=8,
#         learning_rate=0.02,
#         l2_leaf_reg=9,
#         random_seed=2018,
#         metric_period=50,
#         fold_len_multiplier=1.1,
#         loss_function='Logloss',
#         logging_level='Verbose')
#     fine_model = cls.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cates_idx)
#     verbose_feature_importance_cat(fine_model, X_tr)
#     return fine_model


# # In[20]:


# cat = fit_cat(X_tr, y_tr, X_va, y_va, cates_idx)

