
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


def gen_shop_total_da_click(update=False):
    dfal = get_nominal_dfal()
    feat_path = os.path.join(feats_root, 'shop_total_click_da.pkl')
    if os.path.exists(feat_path) and update == False:
        print('Found ' + feat_path)
    else:
        print('Generating ' + feat_path)
        shop_all_click_da = dfal.groupby(['shop_id', 'da'])                                 .size().reset_index()                                 .rename(columns={0: 'agg_shop_total_click_da'})
        dump_pickle(shop_all_click_da, feat_path)
        
    print('gen_shop_total_da_click completed.')


# In[3]:


def gen_shop_da_feature_click(updata=False):
    """生成用户相关所有数据的每天点击统计量"""
    dfal = get_nominal_dfal()
    stats_feat = [
        'item_category_list', 'item_brand_id', 'item_city_id',
        'user_gender_id', 'user_occupation_id', 'item_price_level',
        'item_sales_level', 'item_collected_level', 'item_pv_level',
        'user_age_level', 'user_star_level', 'context_page_id', 'item_id',
        'user_id'
    ]
    tbar = tqdm(stats_feat)
    for feat in tbar:
        feat_path = os.path.join(feats_root, 'shop_' + feat + '_click_da.pkl')
        if os.path.exists(feat_path) and updata == False:
            tbar.set_description('Found {:>60}'.format(os.path.basename(feat_path)))
        else:
            tbar.set_description('Generating {:>60}'.format(
                os.path.basename(feat_path)))
            shop_feat_click_da = dfal.groupby(['shop_id', 'da', feat])                                     .size().reset_index()                                     .rename(columns={0: 'agg_shop_%s_click_da' % feat})
            dump_pickle(shop_feat_click_da, feat_path)

    print('gen_shop_da_feature_click completed.')


# In[4]:


def gen_shop_ho_feature_click(updata=False):
    """生成用户相关所有数据的每天每小时点击统计量"""
    dfal = get_nominal_dfal()
    stats_feat = [
        'item_category_list', 'item_brand_id', 'item_city_id',
        'user_gender_id', 'user_occupation_id', 'item_price_level',
        'item_sales_level', 'item_collected_level', 'item_pv_level',
        'user_age_level', 'user_star_level', 'context_page_id', 'item_id',
        'user_id'
    ]
    tbar = tqdm(stats_feat)
    for feat in tbar:
        feat_path = os.path.join(feats_root, 'shop_' + feat + '_click_ho.pkl')
        if os.path.exists(feat_path) and updata == False:
            tbar.set_description('Found {:>60}'.format(os.path.basename(feat_path)))
        else:
            tbar.set_description('Generating {:>60}'.format(os.path.basename(feat_path)))
            shop_feat_click_ho = dfal.groupby(['shop_id', 'da', 'ho', feat])                                     .size().reset_index()                                     .rename(columns={0: 'agg_shop_%s_click_ho' % feat})
            dump_pickle(shop_feat_click_ho, feat_path)
    print('gen_shop_ho_feature_click completed.')


# In[5]:


def add_shop_total_da_click(data):
    """
    添加用户当天的点击总数
    拼接键['user_id', 'da']
    """
    feat_path = feats_root + 'shop_total_click_da.pkl'
    if not os.path.exists(feat_path):
        gen_shop_total_da_click()
    shop_total_click_da = load_pickle(feat_path)
    data = pd.merge(data, shop_total_click_da, 'left', ['da','shop_id'])
    print('add_shop_total_da_click completed.')
    return data


# In[6]:


def add_shop_da_feature_click(data):
    stats_feat = [
        'item_category_list', 'item_brand_id', 'item_city_id',
        'user_gender_id', 'user_occupation_id', 'item_price_level',
        'item_sales_level', 'item_collected_level', 'item_pv_level',
        'user_age_level', 'user_star_level', 'context_page_id', 'item_id',
        'user_id'
    ]
    
    tbar = tqdm(stats_feat)
    for feat in tbar:
        feat_path = os.path.join(feats_root, 'shop_' + feat + '_click_da.pkl')
        feat_da_click = load_pickle(feat_path)
        tbar.set_description('adding ' + os.path.basename(feat_path))
        data = pd.merge(data, feat_da_click, 'left', [feat, 'da', 'shop_id'])
    print('add_shop_da_feature_click completed.')
    return data


# In[7]:


def add_shop_ho_feature_click(data):
    stats_feat = [
        'item_category_list', 'item_brand_id', 'item_city_id',
        'user_gender_id', 'user_occupation_id', 'item_price_level',
        'item_sales_level', 'item_collected_level', 'item_pv_level',
        'user_age_level', 'user_star_level', 'context_page_id', 'item_id',
        'user_id'
    ]
    
    tbar = tqdm(stats_feat)
    for feat in tbar:
        feat_path = os.path.join(feats_root, 'shop_' + feat + '_click_ho.pkl')
        feat_da_click = load_pickle(feat_path)
        tbar.set_description('adding ' + os.path.basename(feat_path))
        data = pd.merge(data, feat_da_click, 'left', [feat, 'ho', 'da', 'shop_id'])
    print('add_shop_ho_feature_click completed.') 
    return data


# In[8]:


def gen_shop_click_stats(data, col):
    clicks_shop = pd.DataFrame(data.groupby(['shop_id', col])['dt'].count(), )
    clicks_shop.rename(columns={'dt': col+'_m'}, inplace=True)
    clicks_shop.reset_index(inplace=True)
    clicks_shop_avg = pd.DataFrame(clicks_shop.groupby(['shop_id'])[col+'_m'].mean()).rename(columns={col+'_m': col+'_avg'}).reset_index()
    clicks_shop_max = pd.DataFrame(clicks_shop.groupby(['shop_id'])[col+'_m'].max()).rename(columns={col+'_m': col+'_max'}).reset_index()
    clicks_shop_min = pd.DataFrame(clicks_shop.groupby(['shop_id'])[col+'_m'].min()).rename(columns={col+'_m': col+'_min'}).reset_index()
    data = pd.merge(data, clicks_shop_avg, how='left', on='shop_id')
    data = pd.merge(data, clicks_shop_max, how='left', on='shop_id')
    data = pd.merge(data, clicks_shop_min, how='left', on='shop_id')
    print('gen_shop_click_stats {} completed.'.format(col))
    return data


# In[9]:


def add_shop_click_stats(data):
    feat_path = os.path.join(feats_root, 'shop_click_stats.pkl')
    if not os.path.exists(feat_path):
        gen_shop_stats_feature()
    shop_click_stats = load_pickle(feat_path)
    data = pd.merge(data, shop_click_stats, how='left', on='shop_id')
    print('add_shop_click_stats completed.')
    return data


# In[10]:


def gen_shop_stats_feature(updata=False):
    feat_path = os.path.join(feats_root, 'shop_click_stats.pkl')
    if os.path.exists(feat_path) and updata == False:
        print('Found ' + feat_path)
    else:
        dfal = get_nominal_dfal()
        dfal = add_shop_total_da_click(dfal)
        dfal = add_shop_da_feature_click(dfal)
        print('generating ' + feat_path)
        columns_da = list(filter(lambda x: x.endswith('_click_da'), dfal.columns.values))
        columns_ho = list(filter(lambda x: x.endswith('_click_ho'), dfal.columns.values))

        tbar = tqdm(columns_da)
        for col in tbar:
            tbar.set_description('add_shop_click_stats ' + col)
            dfal = gen_shop_click_stats(dfal, col)
        print('gen_shop_click_stats completed.')
        
        feat_names = list(filter(lambda x: '_click_da_' in x, dfal.columns.values))
        dfal = dfal[feat_names + ['shop_id']].drop_duplicates(['shop_id'])
        print('gen_shop_stats_feature shape:', dfal.shape)
        dump_pickle(dfal, feat_path)
    print('gen_shop_stats_feature completed.')


# In[11]:


if __name__ == '__main__':
    gen_shop_total_da_click(False)
    gen_shop_da_feature_click(False)
    gen_shop_ho_feature_click(False)
    gen_shop_stats_feature(False)

