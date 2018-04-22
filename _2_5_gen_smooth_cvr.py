
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
np.random.seed(2018)
import random
import scipy.special as special

from tqdm import tqdm
from utils import (load_pickle, dump_pickle, get_nominal_dfal, feats_root,
                   ordinal_cate_cols, nominal_cate_cols, identity_cols)


# In[2]:


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in tqdm_notebook(range(iter_num)):
            new_alpha, new_beta = self.__fixed_point_iteration(
                imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(
                    new_beta - self.beta) < epsilon:
                break
            #print (new_alpha,new_beta,i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (
                special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) -
                               special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) -
                            special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (
            numerator_beta / denominator)


# In[3]:


def gen_hist_cvr_smooth(start_da, end_da, key, alpha=0.25):
    dfal = get_nominal_dfal()
    dfal = dfal.loc[dfal.da <= end_da, [key, 'da', 'is_trade']]
    gc.collect()
    for da in tqdm(np.arange(start_da, end_da + 1)):
        feat_path = os.path.join(
            feats_root, key + '_hist_cvr_smooth_da_' + str(da) + '.pkl')
        if os.path.exists(feat_path):
            print('found ' + feat_path)
        else:
            print('generating ' + feat_path)
            dfcv =  dfal.copy().loc[dfal.da < da]
            dfcv.is_trade = dfcv.is_trade.apply(int)
            dfcv = pd.get_dummies(dfcv, columns=['is_trade'], prefix='label')
            dfcv = dfcv.groupby([key], as_index=False).sum()
            dfcv[key + '_cvr'] = (dfcv['label_1'] + alpha) / (dfcv['label_0'] + dfcv['label_1'] + alpha * 2)
            result = pd.merge(
                dfal.loc[dfal.da == da, ['da', key]],
                dfcv.loc[:, [key, key + '_cvr']],
                'left',
                on=[key,])
            result.drop_duplicates(['da', key], inplace=True)
            result.sort_values(['da', key], inplace=True)
            dump_pickle(result.loc[:, ['da', key, key + '_cvr']], feat_path)


# In[4]:


def add_hist_cvr_smooth(data, key):
    hist_cvr_smooth = None
    tbar = tqdm(sorted(data.da.unique()))
    for da in tbar:
        feat_path = os.path.join(feats_root, key + '_hist_cvr_smooth_da_' + str(da) + '.pkl')
        tbar.set_description('adding hist cvr smooth {},{}'.format(key, da))
        da_cvr_smooth = load_pickle(feat_path)
        if hist_cvr_smooth is None:
            hist_cvr_smooth = da_cvr_smooth
        else:
            hist_cvr_smooth = pd.concat([hist_cvr_smooth, da_cvr_smooth], axis=0)
    data = pd.merge(data, hist_cvr_smooth, 'left', ['da', key])
    print('add_hist_cvr_smooth {} completed'.format(key))
    return data


# In[5]:


if __name__ == '__main__':
    for c in tqdm(ordinal_cate_cols + nominal_cate_cols + identity_cols + ['hm','mi','ho'] ):
        gen_hist_cvr_smooth(18, 24, c)

