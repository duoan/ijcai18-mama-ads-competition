import os
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import catboost as cb
# import xgboost as xg
import lightgbm as lg
import operator

input_root = './input/'
feats_root = './feats/'
cache_root = './cache/'
model_root = './model/'
rests_root = './rests/'


nominal_cate_cols = [
    'item_category_list', 'item_brand_id', 'item_city_id', 'user_gender_id',
    'user_occupation_id'
]

ordinal_cate_cols = [
    'item_price_level', 'item_sales_level', 'item_collected_level',
    'item_pv_level', 'user_age_level', 'user_star_level', 'context_page_id',
    'shop_review_num_level', 'shop_star_level'
]

identity_cols = ['item_id', 'shop_id', 'user_id']

continual_cols = [
    'shop_review_positive_rate', 'shop_score_delivery',
    'shop_score_description', 'shop_score_service'
]

datetime_cols = ['dt', 'ts', 'da', 'ho', 'hm', 'mi']

textual_cols = ['item_property_list', 'predict_category_property']


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s%%' % n
    return percentile_


def top(x):
    return x.value_counts().index[0]


def freq(x):
    return x.value_counts().values[0]


def unique(x):
    return len(np.unique(x))


def vrange(x):
    return np.max(x) - np.min(x)


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def reduce_mem_usage(data, ignore_cols=['is_trade']):
    data = data.copy()
    start_mem_usg = data.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in data.columns:
        if col in ignore_cols:
            continue
        dtype = data[col].dtype
        # print(dtype)
        if dtype in [np.int, np.float]:

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", data[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = data[col].max()
            mn = data[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(data[col]).all():
                NAlist.append(col)
                data[col].fillna(data[col].mean(), inplace=True)

            # test if column can be converted to an integer
            asint = data[col].fillna(0).astype(np.int64)
            result = (data[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        data[col] = data[col].astype(np.uint8)
                    elif mx < 65535:
                        data[col] = data[col].astype(np.uint16)
                    elif mx < 4294967295:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                data[col] = data[col].astype(np.float16)

            # Print new column type
            print("dtype after: ", data[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = data.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return data, NAlist


def level_enc(df, field):
    df = df.copy()
    levels = sorted(df[field].unique())
    df[field] = df[field].apply(lambda x: levels.index(x))
    return df


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def dump_pickle(obj, path, protocol=None):
    pickle.dump(obj, open(path, 'wb'), protocol=protocol)


def get_nominal_dfal():
    return load_pickle(os.path.join(cache_root, 'dfda_nominal.pkl'))


def get_textual_dfal():
    return load_pickle(os.path.join(cache_root, 'dfda_textual.pkl'))


def add_time_fields(df, copy=True):
    df_ = df.copy()
    # 时间迁移16个小时，正好可以按天划分数据
    df_['dt'] = pd.to_datetime(df_.context_timestamp, unit='s') - timedelta(hours=16)
    df_['ts'] = df_.context_timestamp
    df_['da'] = df_['dt'].apply(lambda x: x.day)
    df_['ho'] = df_['dt'].apply(lambda x: x.hour)
    df_['hm'] = df_['dt'].apply(lambda x: int(x.strftime('%H%M')))
    df_['mi'] = df_['dt'].apply(lambda x: int(x.strftime('%M')))
    return df_


def print_feature_importance_lgb(gbm):
    print(80 * '*')
    print(31 * '*' + 'Feature Importance' + 31 * '*')
    print(80 * '.')
    print("\n".join((".%50s => %9.5f" % x) for x in sorted(
        zip(gbm.feature_name(), gbm.feature_importance("gain")),
        key=lambda x: x[1],
        reverse=True)))
    print(80 * '.')


def fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols):
    params = {
        'max_depth': 8,
        'num_leaves': 128,
        'objective': 'binary',
        'min_data_in_leaf': 20,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'subsample': 0.85,
        'bagging_freq': 1,
        'random_state': 2018,
        'metric': ['binary_logloss'],
        'num_threads': 16,
    }

    MAX_ROUNDS = 10000
    dtr = lg.Dataset(X_tr, label=y_tr, categorical_feature=cates_cols)
    dva = lg.Dataset(X_va, label=y_va, categorical_feature=cates_cols, reference=dtr)

    cls = lg.train(
        params,
        dtr,
        num_boost_round=MAX_ROUNDS,
        valid_sets=(dva, dtr),
        valid_names=['valid', 'train'],
        early_stopping_rounds=125,
        verbose_eval=50)
    print_feature_importance_lgb(cls)
    lg.plot_importance(cls, importance_type='gain', figsize=(11, 12), max_num_features=50, grid=False)
    return cls


def verbose_feature_importance_cat(cls, X_tr):
    cat_feature_importance = {
        X_tr.columns.values.tolist()[idx]: score
        for idx, score in enumerate(cls.feature_importances_)
    }

    cat_feature_importance = sorted(cat_feature_importance.items(),
                                    key=operator.itemgetter(1),
                                    reverse=False)

    print(80 * '*')
    print(31 * '*' + 'Feature Importance' + 31 * '*')
    print(80 * '.')
    for feature, score in reversed(cat_feature_importance):
        print(".%50s => %9.5f" % (feature, score))
    print(80 * '.')

    feature_score = pd.DataFrame(cat_feature_importance, columns=['Feature', 'Score'])

    plt.rcParams["figure.figsize"] = (11, 12)
    ax = feature_score.tail(50).plot('Feature', 'Score', kind='barh', color='b')
    ax.set_title("Catboost Feature Importance Ranking", fontsize=8)
    ax.set_xlabel('')
    rects = ax.patches
    # get feature score as labels round to 2 decimal
    labels = feature_score.tail(50)['Score'].round(2)
    for rect, label in zip(rects, labels):
        width = rect.get_width()
        ax.text(width + 0.2, rect.get_y() + 0.02, label, ha='center', va='bottom')
    plt.show()


def fit_cat(X_tr, y_tr, X_va, y_va, cates_idx):
    print('Fitting CatBoostClassifier ...')
    cls = cb.CatBoostClassifier(
        iterations=2000,
        od_type='Iter',
        od_wait=120,
        max_depth=8,
        learning_rate=0.02,
        l2_leaf_reg=9,
        random_seed=2018,
        metric_period=50,
        fold_len_multiplier=1.1,
        loss_function='Logloss',
        logging_level='Verbose')
    cls = cls.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cates_idx)
    verbose_feature_importance_cat(cls, X_tr)
    return cls
