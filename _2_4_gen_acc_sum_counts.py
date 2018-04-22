
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle,dump_pickle, get_nominal_dfal, feats_root


# In[2]:


def gen_id_global_sum_count(last_da=23,
                            stats_feats=[
                                'item_id', 'shop_id', 'user_id',
                                'item_brand_id', 'item_city_id', 'hm'
                            ]):
    dfal = get_nominal_dfal()
    dfal = dfal.loc[dfal.da < last_da, stats_feats]
    for feat in tqdm(stats_feats):
        feat_path = os.path.join(feats_root,'global_count_' + feat + '_lastda' + str(last_da) + '.pkl')
        if os.path.exists(feat_path):
            print('found ' + feat_path)
        else:
            print('generating ' + feat_path)
            feat_count_sum = pd.DataFrame(
                dfal.groupby(feat).size()).reset_index().rename(
                    columns={0: 'agg_' + feat + '_sum_count'})
            dump_pickle(feat_count_sum, feat_path)


# In[3]:


def add_global_count_sum(data,
                         last_da=23,
                         stats_feats=[
                             'item_id', 'shop_id', 'user_id', 'item_brand_id',
                             'item_city_id'
                         ]):
    """
    添加ID出现次数，根据ID_name拼接
    """
    for feat in tqdm(stats_feats):
        feat_path = os.path.join(
            feats_root,
            'global_count_' + feat + '_lastda' + str(last_da) + '.pkl')
        if not os.path.exists(feat_path):
            gen_id_global_sum_count(last_da, [feat])
        feat_count_sum = load_pickle(feat_path)
        data = data.merge(feat_count_sum, 'left', [feat])
    return data


# In[4]:


if __name__ =='__main__':
    gen_id_global_sum_count(24)
    gen_id_global_sum_count(23)
    gen_id_global_sum_count(22)
    gen_id_global_sum_count(21)
    gen_id_global_sum_count(20)
    gen_id_global_sum_count(19)
    gen_id_global_sum_count(18)
    print('all done')

