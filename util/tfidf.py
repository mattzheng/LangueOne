# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import copy

def docs(w, D):
	'''
		w,词;
		D,整个文档，分好词的，譬如[('你','好'),('我','们')....]
		计算含有w词的文档数量
	'''
	c = 0
	for d in D:
		if w in d:
			c = c + 1;
	return c


def tfidf(jieba_content,min_count = 5):
    '''
    作用：生成、计算idf
    参考：http://www.voidcn.com/article/p-mhebqvic-qq.html
    输入：分行的文本内容;
    输出：dataframe,分别有:words /  tf / idf  三列
    '''
#    try:
#        diction = pd.DataFrame( list(set(sum(jieba_content,[])) ) , columns = ['word'])
#    except:
#        diction = pd.DataFrame( list(set(sum(jieba_content,())) ) , columns = ['word'])
    print('Calculate TF ...')
    jieba_content_ = []
    [jieba_content_.extend(i) for i in jieba_content]
    diction = pd.DataFrame( jieba_content_ , columns = ['word'])
    diction = diction['word'].value_counts()
    diction = pd.DataFrame(diction)
    diction.columns = ['tf']
    diction['word'] = diction.index
    diction.reset_index(inplace=True,drop=True)
    # 过滤低频词汇
    diction = diction[diction['tf']>min_count]
    diction.reset_index(inplace=True,drop=True)
    print('Calculate DF ...')
    df = []
    for x in tqdm(diction['word']):
        df.append(docs(x,jieba_content))
    diction['df'] =  df

    print('Calculate IDF ...')
    n = len(jieba_content)
    idf = []
    for i in tqdm(n*1.0 / (diction['tf'] + 1 )):
        idf.append(math.log(i)) 
    diction['idf'] = idf
    diction['tfidf'] = diction['tf'] * diction['idf']
    #diction['idf'] = [math.log(i) for i in   n*1.0 / (diction['tf'] + 1 ) ]
    return diction


if __name__ == '__main__': 
    print(tfidf([[1,2,3,4,5,6],[3,5,2,8,9]],min_count = 0))






