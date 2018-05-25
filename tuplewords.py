# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:53:15 2018

@author: mattzheng
"""

# 8536 55.45min
import itertools
from tqdm import tqdm
import pandas as pd

class TupleWords(object):
    def __init__(self,stop_word = None):
        self.stop_word = stop_word
    
    def Tuples(self,corpus):
        # 输入list,[a,b,c]
        # 输出：list,[[a,b],[a,c],[b,c]]
        tuple_corpus = []
        for i in tqdm(itertools.combinations(corpus, 2)):
            tuple_corpus.append(list(i))
        return tuple_corpus

    def DeleteStopword(self,stop_word,data_list):
        print('Delete Stopwords ... ')
        return [ list(set(i)  - set(stop_word)) for i in data_list]

    def CoOccurrence(self,data_list):
        '''
        输入：输入,list,[[a,b],[b,c]]
        
        输出：
            id_pools = [a,b,c]
            tuple_pools = ['a,b':1,'a,c':1,'b,c':1]
        '''
        id_pools = []
        tuple_pools = {}
        if self.stop_word:
            data_list = self.DeleteStopword(self.stop_word,data_list)
        
        print('Calculate Word Co-occurrence...')
        for i in tqdm(range(len(data_list)),total = len(data_list)):
            if len(data_list[i]) > 1:
                # 如果一句话只有一个词，就pass
                one_tuple = self.Tuples(data_list[i])
                # 
                for one in one_tuple:
                    one_id = ','.join(set(one))
                    if one_id not in id_pools:
                        tuple_pools[one_id] = 1
                        id_pools.append(one_id)
                    else:
                        tuple_pools[one_id] += 1
        return id_pools,tuple_pools

    # 内容变成dataframe
    def tansferDataFrame(self,tuple_pools):
        '''
        输入：tuple_pools = ['a,b':1,'a,c':1,'b,c':1]
        
        输出：dataframe格式，a,b,1
                             a,c,1
                             b,c,1
        
        '''
        word_x,word_y,freq = [],[],[]
        print('tansfer dataframe ... ')
        for i,j in tqdm(tuple_pools.items()):
            word = i.split(',')
            word_x.append(word[0])
            word_y.append(word[1])
            freq.append(j)

        CoOccurrence_data = pd.DataFrame({'word_x':word_x,'word_y':word_y,'freq':freq})
        CoOccurrence_data =  CoOccurrence_data.sort_index(by='freq',ascending=False) 
        return CoOccurrence_data

    # 热词统计模块
    def Hotwords(self,data_list):
        hotwords = sum(data_list,[])
        hotwords_dataframe = pd.Series(hotwords).value_counts()  
        hotwords_dataframe = pd.DataFrame({'words':hotwords_dataframe.keys(),'freq':hotwords_dataframe})
        return hotwords_dataframe

# 热词 - 引文标签结合
    def get_hotwords2article(self,hotwords_dataframe,maindata):

        word_list = []
        word_length = []
        print('hotwords add article ...')
        for word in tqdm(hotwords_dataframe.words):
            con = list(maindata.index[[word in i for i in maindata.keyword]])
            word_list.append(  con  )
            word_length.append(len(con))

        hotwords_dataframe['id'] = word_list
        hotwords_dataframe['len'] = word_length
        return hotwords_dataframe

#
if __name__ == '__main__':
    # 二元组
# 时间：8536/51min
    data = pd.read_csv('toutiao_data.csv',encoding = 'utf-8')


def not_nan(obj):
    return obj == obj

keywords = []
for word in tqdm(data.new_keyword):
    if not_nan(word):
        keywords.append(word.split(','))
        
        
stop_word = ['方法','结论']
tw = TupleWords(stop_word)

id_pools,tuple_pools = tw.CoOccurrence(keywords[:1000])
    
# 内容变成dataframe
CoOccurrence_data = tw.tansferDataFrame(tuple_pools)
CoOccurrence_data
        
# 热词统计模块
hotwords_dataframe = tw.Hotwords(keywords)
hotwords_dataframe

