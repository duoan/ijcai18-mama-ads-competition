
# coding: utf-8

# In[1]:


import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import utils


# In[2]:


def gen_dfal():
    dump_nominal_file = os.path.join(utils.cache_root, 'dfda_nominal.pkl')
    dump_textual_file = os.path.join(utils.cache_root, 'dfda_textual.pkl')
    if not os.path.exists(dump_nominal_file):
        tr = pd.read_csv('./input/round1_ijcai_18_train_20180301.txt', sep=' ', dtype={'is_trade':np.uint8})
        tr.is_trade = tr.is_trade.astype(np.int8)
        te = pd.read_csv('./input/round1_ijcai_18_test_b_20180418.txt', sep=' ')
        da = pd.concat([tr, te], axis=0)
        da = utils.add_time_fields(da)
        
        for col in utils.nominal_cate_cols + utils.identity_cols:
            da[col] = LabelEncoder().fit_transform(da[col])
        
        for col in utils.ordinal_cate_cols:
            levels = sorted(da[col].unique())
            da[col] = da[col].apply(lambda x : levels.index(x)).astype(np.uint8)
        
        del da['context_id']
        del da['context_timestamp']
        del da['ts']
        da, _ = utils.reduce_mem_usage(da)
        utils.dump_pickle(da[utils.textual_cols], dump_textual_file)
        utils.dump_pickle(da.drop(utils.textual_cols, axis=1), dump_nominal_file)
    print('gen dfal ok.')


# In[3]:


if __name__ == '__main__':
    gen_dfal()

