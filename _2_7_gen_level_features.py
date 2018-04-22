
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import (load_pickle, dump_pickle, get_nominal_dfal, feats_root,
                   mem_usage, reduce_mem_usage, nominal_cate_cols,
                   ordinal_cate_cols, identity_cols, continual_cols, top, freq,
                   unique, vrange, percentile)

pd.set_option('display.max_columns', 1000)


# In[2]:


level_cols = [
    'item_price_level', 'item_sales_level', 'item_collected_level',
    'item_pv_level', 'user_age_level', 'user_star_level',
    'shop_review_num_level', 'shop_star_level'
]


# In[3]:


def gen_level_agg_features(data, last_da, win_das, col):
    agg_cols = list(filter(lambda x: not x.startswith(col[:4]), level_cols))
    data = data.copy()
    indexing = (data.da < last_da) & (data.da >= last_da - win_das)
    gp = data.loc[indexing, [col] + agg_cols].groupby(col)[agg_cols]

    aggs = gp.agg([
        'mean', 'std', 'sem', pd.DataFrame.kurt, pd.DataFrame.skew,
        pd.DataFrame.mad, freq,
        percentile(.3),
        percentile(.9)
    ])
    aggs.columns = [
        'agg_level_{}_{}_{}_wd_{}'.format(col, c[0], c[1], win_das)
        for c in aggs.columns
    ]
    aggs = aggs.reset_index()
    data = data.loc[data.da == last_da].merge(aggs, how='left', on=col)
    data.drop(level_cols, inplace=True, axis=1)
    data.drop_duplicates([col, 'da'], inplace=True)
    data.fillna(0, inplace=True)
    data, _ = reduce_mem_usage(data)
    return data


# In[8]:


def gen_level_aggs(col, updata=False):
    feat_path = os.path.join(feats_root,'level_aggs_{}.pkl'.format(col))
    if os.path.exists(feat_path) and updata == False:
        print('Found ' + feat_path)
    else:
        print('Generating ' + feat_path)
        dfal = get_nominal_dfal()[[col, 'da'] + level_cols]
        dmax = dfal.da.max()
        dmin = dfal.da.min()
        
        level_agg = None
        for da in sorted(dfal.da.unique())[1:]:
            da_agg = None
            for win_das in [1, 2, 3]:
                if da - win_das < dmin:
                    continue
                agg = gen_level_agg_features(dfal, da, win_das, col)
                print('Generated {} {} {}'.format(col, da, win_das))
                if da_agg is None:
                    da_agg = agg
                else:
                    da_agg = da_agg.merge(agg, how='outer')
            if level_agg is None:
                level_agg = da_agg
            else: 
                level_agg = pd.concat([level_agg, da_agg], axis=0)
                level_agg.fillna(0, inplace=True)
                level_agg, _ = reduce_mem_usage(level_agg)
        print(level_agg.shape)
        level_agg, _ = reduce_mem_usage(level_agg)
        dump_pickle(level_agg, feat_path)


# In[13]:


def gen_level_features():
    for c in tqdm(['item_id','shop_id','user_id','item_brand_id','item_city_id','hm', 'mi', 'ho']):
        gen_level_aggs(c)


# In[14]:


def add_level_features(data, col):
    feat_path = os.path.join(feats_root,'level_aggs_{}.pkl'.format(col))
    if not os.path.exists(feat_path):
        gen_level_features(col)
    agg = load_pickle(feat_path)
    return pd.merge(data, agg, how='left',on=[col, 'da'])


# In[15]:


if __name__ == '__main__':
    gen_level_features()


# In[ ]:


# dfal = get_nominal_dfal()


# In[ ]:


# dfal.shape


# In[ ]:


# dfal = dfal.loc[dfal.da>20,:]


# In[ ]:


# for c in tqdm_notebook(nominal_cate_cols + ordinal_cate_cols + identity_cols):
#     dfal = add_target_features(dfal, c)


# In[ ]:


# del dfal['dt']
# for c in dfal.columns:
#     if c.endswith('_wd_6'):
#         del dfal[c]


# In[ ]:


# dfal, _ = reduce_mem_usage(dfal)


# In[ ]:


# dfal.columns.values


# In[ ]:


# X_tr = dfal.loc[dfal.da<=22,:].drop(['da', 'hm', 'instance_id', 'is_trade'] + identity_cols, axis=1)
# y_tr = dfal.loc[dfal.da<=22,'is_trade']
# X_va = dfal.loc[dfal.da==23,:].drop(['da', 'hm', 'instance_id', 'is_trade'] + identity_cols, axis=1)
# y_va = dfal.loc[dfal.da==23,'is_trade']


# In[ ]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# import catboost as cb
# import xgboost as xg
# import lightgbm as lg


# In[ ]:


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


# In[ ]:


# gbm = fit_lgb(X_tr, y_tr, X_va, y_va, nominal_cate_cols)


# ## CatBoostClassifier

# In[ ]:


# cates_idx = [X_tr.columns.values.tolist().index(c) for c in nominal_cate_cols]


# In[ ]:


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


# In[ ]:


# cat = fit_cat(X_tr, y_tr, X_va, y_va, cates_idx)

