# 天池阿里妈妈搜索广告初赛方案

[比赛介绍](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231647)

### 依赖包
sklearn、lightgbm、catboost、pandas、numpy、matplotlib、pickle、h5py、tqdm

### 运行环境
jupyter + python3.5

### 运行准备
创建目录：```mkdir input && mkdir cache && mkdir feats && mkdir rests```

### 数据预处理：
添加时间维度特征合并数据后存为pkl格式。[源码](_1_1_pre_process_dfal.ipynb)

### 特征工程：
- 1、通用点击率 [user源码](./_2_1_gen_user_features.ipynb) [item源码](./_2_2_gen_item_features.ipynb) [shop源码](./_2_3_gen_shop_features.ipynb)
- 2、各维度时间加权后的点击量 [源码](./_2_4_gen_acc_sum_counts.ipynb)
- 3、平滑后的CVR [源码](./_2_5_gen_smooth_cvr.ipynb)
- 4、target特征处理，（均值、方差、标准差等描述性统计特征）[源码](./_2_6_gen_target_features.ipynb)
- 5、item、user、shop维度下各个原始level特征（如item_price_level）的描述性统计特征 [源码](./_2_7_gen_level_features.ipynb)
- 6、user最后1~2次的行为特征 [源码](./_2_8_gen_last_features.ipynb)

### 模型
使用了LigthGBM和CatBoost(训练太慢，performance很好)，因为自己懒，这两个模型都可以很好的自动处理类别型特征。每个算法训练2个模型，第2个模型是删除掉在第一个模型评估出不重要的特征进行训练。 具体见[源码](./_3_1_gen_final_dataset.ipynb)

### 总结
感谢组委会提供这么好的竞赛机会通大神们一起学习。虽然是自己第一次参加这类数据竞赛成绩也不是很理想，不过真实的学到了很多知识，有读了很多paper，CTR预估模型千千万（LR、GBDT、Wide&Deep、FM&FFM、DeepFM...），但是后来还是只用了简单的模型，精力太有限了。我把所有的功夫都花在了特征处理上，如果没有好的特征和数据处理喂进去，即使再牛逼的模型也是白瞎。关于特征，一开始我就用原始的特征，然后一点点加进去，看feature重要度，然后再根据重要度进行延伸，同时也考虑特征的多样性。用模型进行验证后去掉没有用的特征，不断迭代。

### 相关链接
- [catboost & lightgbm & xgboost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)
- [从ctr预估问题看看f(x)设计—DNN篇](https://zhuanlan.zhihu.com/p/28202287)
- [Kaggle实战——点击率预估](https://zhuanlan.zhihu.com/p/32500652)
- [Python target encoding for categorical features
](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)
- [pandas的shift和diff介绍，不同行（列）移动，做差](https://blog.csdn.net/lz_peter/article/details/79109185)
