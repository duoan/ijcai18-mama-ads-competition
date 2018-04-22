
# coding: utf-8

# In[1]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[1]:


import os
import pickle
import gc
import pandas as pd
pd.set_option('display.max_columns', 1000)
import numpy as np
from tqdm import tqdm_notebook, tnrange, tqdm
from utils import (reduce_mem_usage, load_pickle, dump_pickle, get_nominal_dfal, feats_root,
                   ordinal_cate_cols, nominal_cate_cols, identity_cols,
                   fit_cat, fit_lgb, verbose_feature_importance_cat)

import warnings
warnings.filterwarnings(action='ignore')

import h5py


# In[2]:


from _2_1_gen_user_features import (
    add_user_click_stats, add_user_da_feature_click, add_user_ho_feature_click,
    add_user_total_da_click)
from _2_2_gen_item_features import (
    add_item_click_stats, add_item_da_feature_click, add_item_ho_feature_click,
    add_item_total_da_click)
from _2_3_gen_shop_features import (
    add_shop_click_stats, add_shop_da_feature_click, add_shop_ho_feature_click,
    add_shop_total_da_click)
from _2_4_gen_acc_sum_counts import add_global_count_sum
from _2_5_gen_smooth_cvr import add_hist_cvr_smooth
from _2_6_gen_bagging_features import add_target_features
from _2_7_gen_level_features import add_level_features


# In[3]:


def gen_final_dataset(tr_start_da, tr_end_da, te_da=24):
    tr_dump_file = './cache/final_dataset_tr_{}_{}.h5'.format(tr_start_da, tr_end_da)
    te_dump_file = './cache/final_dataset_te_{}.h5'.format(te_da)
    
    dftr = None
    dfte = None
    if os.path.exists(tr_dump_file):
        print('Found ' + tr_dump_file)
        store = pd.HDFStore(tr_dump_file, mode='r',complevel=9,)
        dftr = store['dataset']
        store.close()
    elif dftr is None:
        dfal = get_nominal_dfal()
        dftr = dfal.loc[(dfal.da >= tr_start_da) & (dfal.da <= tr_end_da)]
        print('Generating Train Dataset...')
        ##################################################################
        # add user click
        #dftr = add_user_click_stats(dftr)
        #dftr = add_user_total_da_click(dftr)
        #dftr = add_user_da_feature_click(dftr)
        #dftr = add_user_ho_feature_click(dftr)
        # add item click
        #dftr = add_item_click_stats(dftr)
        #dftr = add_item_total_da_click(dftr)
        #dftr = add_item_da_feature_click(dftr)
        #dftr = add_item_ho_feature_click(dftr)
        # add shop click
        #dftr = add_shop_click_stats(dftr)
        #dftr = add_shop_total_da_click(dftr)
        #dftr = add_shop_da_feature_click(dftr)
        #dftr = add_shop_ho_feature_click(dftr)
        # add global count sum
        dftr = add_global_count_sum(dftr, tr_end_da)

        # add smooth cvr
        for c in tqdm(ordinal_cate_cols + nominal_cate_cols + identity_cols + ['hm','mi'], desc='add_hist_cvr_smooth'):
            dftr = add_hist_cvr_smooth(dftr, c)
        print('add_hist_cvr_smooth completed')
        
        #for c in tqdm(['item_id', 'shop_id','user_id', 'item_brand_id','item_city_id','hm', 'mi'], desc='add_target_features'):
        #    dftr = add_target_features(dftr, c)
        #print('add_target_features completed')
        
        # for c in tqdm(nominal_cate_cols + ['hm', 'mi', 'ho'], desc='add_level_features'):
        #    dftr = add_level_features(dftr, c)
            
        print('add_level_features completed')
        print(dftr.shape)
        store = pd.HDFStore(tr_dump_file, mode='w',complevel=9)
        store['dataset'] = dftr
        store.close()
        del dfal
        gc.collect()
        print('Generated Train Dataset')
        
    if os.path.exists(te_dump_file):
        print('Found ' + te_dump_file)
        store = pd.HDFStore(te_dump_file, mode='r',complevel=9)
        dfte = store['dataset']
        store.close()
    elif dfte is None:
        dfal = get_nominal_dfal()
        dfte = dfal.loc[dfal.da == te_da]
        ##################################################################
        print('Generating Test Dataset...')
        # add user click
        #dfte = add_user_click_stats(dfte)
        #dfte = add_user_total_da_click(dfte)
        #dfte = add_user_da_feature_click(dfte)
        #dfte = add_user_ho_feature_click(dfte)
        # add item click
        #dfte = add_item_click_stats(dfte)
        #dfte = add_item_total_da_click(dfte)
        #dfte = add_item_da_feature_click(dfte)
        #dfte = add_item_ho_feature_click(dfte)
        # add shop click
        #dfte = add_shop_click_stats(dfte)
        #dfte = add_shop_total_da_click(dfte)
        #dfte = add_shop_da_feature_click(dfte)
        #dfte = add_shop_ho_feature_click(dfte)
        # add global count sum
        dfte = add_global_count_sum(dfte, te_da)

        # add smooth cvr
        for c in tqdm(ordinal_cate_cols + nominal_cate_cols + identity_cols + ['hm','mi'], desc='add_hist_cvr_smooth'):
            dfte = add_hist_cvr_smooth(dfte, c)
        print('add_hist_cvr_smooth completed')
        
        #for c in tqdm(['item_id','shop_id','user_id', 'item_brand_id','item_city_id','hm', 'mi'], desc='add_target_features'):
        #    dfte = add_target_features(dfte, c)
        #print('add_target_features completed')
        
        # for c in tqdm(['item_id','shop_id','user_id','item_brand_id','item_city_id', 'hm', 'mi', 'ho'], desc='add_level_features'):
        #     dfte = add_level_features(dfte, c)
        # print('add_level_features completed')
        print(dfte.shape)
        store = pd.HDFStore(te_dump_file, mode='w',complevel=9)
        store['dataset'] = dfte
        store.close()
        
        del dfal
        gc.collect()
        print('Generated Test Dataset')
    #dftr.drop(unused_cols, axis=1, inplace=True)
    #dfte.drop(unused_cols, axis=1, inplace=True)
    return dftr, dfte


# In[4]:


ignore_cols = ['instance_id', 'dt', 'da', 'user_id', 'item_id', 'shop_id', 'item_brand_id']
cates_cols = [
    'item_category_list', 'item_city_id', 'user_gender_id',
    'user_occupation_id'
]


def get_dataset():

    dftr, dfte = gen_final_dataset(19, 23, 24)
    trset = dftr.loc[(dftr.da > 18) & (dftr.da <= 22), :].drop(
        ignore_cols, axis=1)
    vaset = dftr.loc[dftr.da == 23, :].drop(ignore_cols, axis=1)
    teset = dfte.loc[dfte.da == 24, :].drop(ignore_cols, axis=1)

    del dftr
    del dfte
    gc.collect()

    X_tr = trset.drop('is_trade', axis=1)
    X_va = vaset.drop('is_trade', axis=1)
    X_te = teset.drop('is_trade', axis=1)
    y_tr = trset.is_trade
    y_va = vaset.is_trade

    del trset
    del vaset
    del teset
    gc.collect()
    return X_tr, y_tr, X_va, y_va, X_te


# In[5]:


X_tr, y_tr, X_va, y_va, X_te =  get_dataset()


# In[6]:


X_tr.shape,y_tr.shape, X_va.shape, y_va.shape, X_te.shape


# In[ ]:


lgb = fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols)


# In[ ]:


# unimportant_features = []
# for x in sorted(zip(lgb.feature_name(), lgb.feature_importance("gain")), key=lambda x: x[1]):
#     if x[1]<10:
#         unimportant_features.append(x[0])
#
# X_tr.drop(unimportant_features, axis=1, inplace=True)
# X_va.drop(unimportant_features, axis=1, inplace=True)
# X_te.drop(unimportant_features, axis=1, inplace=True)
# cates_cols = list(filter(lambda x : x in X_tr.columns.values.tolist(), cates_cols))
# cates_cols
# lgb = fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols)


# In[ ]:


cates_idx = [X_tr.columns.values.tolist().index(c) for c in cates_cols]


# In[ ]:


cat = fit_cat(X_tr, y_tr, X_va, y_va,[])


# In[ ]:


for score, name in sorted(zip(cat.feature_importances_ , X_tr.columns), reverse=True):
    if score == 0:
        del X_tr[name]
        del X_va[name]
        del X_te[name]
        print('{}: {}'.format(name, score))


# In[ ]:


import catboost as cb
best_cat_params = cat.get_params().copy()
best_cat_params.update({
    'use_best_model': True
})
best_cat = cb.CatBoostClassifier(**best_cat_params)
best_cat.fit(X_tr,y_tr, eval_set=(X_va,y_va))


# In[ ]:


verbose_feature_importance_cat(best_cat,X_tr)


# In[ ]:


# 下一步生成更多的hm相关特征


# In[ ]:


y_te_hat = best_cat.predict_proba(X_te)[:,1]


# In[ ]:


len(y_te_hat)


# In[ ]:


sub = daset.loc[daset.da==24, ['instance_id']]


# In[ ]:


sub.head()


# In[ ]:


sub['predicted_score'] = y_te_hat


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('./rests/20180420-0.0810213-cat.txt', index=False, header=True, sep=' ')


# In[ ]:


# 18372

