# -*- coding: utf-8 -*-

'''
## kaggle借鉴代码（1）——文本多分类下词条重要性

参考：url
https://www.kaggle.com/tks0123456789/word-phrase-importance

思路：

先得到词条/单词的tfidf矩阵，然后通过向量去监督训练（使用的是LR模型），然后根据模型的回归系数作为权重。

启迪:

词条的tfidf代表词语对于整个文本的重要性；

这种方式可以得到词条对于分类的重要性

'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

def get_feature_importances(model,train_data_df_new_4,categories,sort_values_class = 'dish_look', \
                            analyzer = 'char_wb', ngram = 1, lowercase = False,\
                            min_df=10,max_df = 0.9, sampsize=40000):
    '''
    model - 分类模型
    train_data_df_new_4 - dataframe格式
    categories - dataframe要分析的类目
    sort_values_class = 'dish_look' - 选中某个类目,排序
    analyzer = 'char_wb' - TfidfVectorizer的参数,专门为单字,或者'word'
    ngram = 1 - TfidfVectorizer的参数,(1,ngram)
    lowercase = False
    min_df=10
    max_df = 0.9
    sampsize=40000  - 样本随机抽40000
    
    其中,TfidfVectorizer参数可见：https://blog.csdn.net/sinat_26917383/article/details/71436563
    
    return dataframe
    '''
    tfv = TfidfVectorizer(min_df=min_df,
                          strip_accents='unicode',
                          analyzer=analyzer,
                          ngram_range=(1, ngram),
                          max_df=max_df)
    df_sample = train_data_df_new_4.sample(sampsize, random_state=123) # 随机抽样其中的一部分
    X = tfv.fit_transform(df_sample.content)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    terms = tfv.get_feature_names()
    #print('#terms:', len(terms))
    var_imp = pd.DataFrame(index=terms)
    for category in categories:
        y = df_sample[category].values
        model.fit(X, y)
        var_imp[category] =  np.sqrt(scaler.var_) * model.coef_[0]
    var_imp = var_imp.sort_values(sort_values_class, ascending=False)
    return var_imp

# 单词粒度
model = LogisticRegression()
train_data_df_new_4 = pickle.load(open('train_data_df_new_4.pkl', 'rb'))

get_feature_importances(model,train_data_df_new_4,train_data_df_new_4.columns[2:] ,\
                        sort_values_class = 'location_distance_from_business_district',\
                        analyzer = 'char_wb',ngram=3, lowercase=True, min_df=10, \
                        sampsize=50)
