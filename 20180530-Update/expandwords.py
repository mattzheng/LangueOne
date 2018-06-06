#! -*- coding:utf-8 -*-

from tqdm import tqdm
import copy
import pandas as pd
import gensim, logging

def refresh_words(save_words):
    return list(set(save_words))

def augment_word(list_words,topn = 10):
    save_words = []
    new_words = []
    for word in list_words:
        try:
            save_words.extend([i[0] for i in model.most_similar(word,topn = topn)])
        except:
            new_words.append(word)
    return save_words,new_words

def augment_word_loop(list_words,epoch,topn = 10):
    save_words = []
    for _ in tqdm(list(range(epoch))):
        tmp_lists,new_words = augment_word(list_words,topn = topn)
        if tmp_lists:
            list_words = refresh_words(tmp_lists)
            save_words.extend(list_words)
            print('The words have :%s'%len(save_words))
    return save_words,new_words

def which(bool_object):
    '''
    输入：
        bool值
    输出：
        输出true的位置信息
    '''
    i = 0
    number = []
    for j in bool_object:
        i += 1
        if j == True:
            number.append(i-1)
    return(number)

'''
-------------------- 拓词阶段 -------------------- 
不能写成这样：
augment_word_loop('公众号',epoch,topn = topn)
    因为这样就会只提取第一个'公'字
'''

def delete_word(list_funs):
    delete_words = model.doesnt_match(' '.join(list_funs).split())
    print('========delete words : %s ========'%delete_words)
    num = [i for i,j in enumerate(list_funs) if j in delete_words  ]
    for n in num:
        del list_funs[n]
    return list_funs

def delete_word_loop(list_funs,epoch = 2):
    '''
    list_funsss = delete_word_loop(wordlist['公众号'],epoch = 10)
    print(len(wordlist['公众号']))
    '''
    tmp_list_words = copy.copy(list_funs)
    for _ in range(epoch):
        tmp_list_words = delete_word(tmp_list_words)
    return tmp_list_words

def dict2dataframe(wordlist):
    word_dataframe = pd.DataFrame()
    for keys,values in tqdm(wordlist.items()):
        word_dataframe = pd.concat([word_dataframe,pd.DataFrame({'words':keys,'sim_words':values})],ignore_index=False)
    return word_dataframe

def ExpandWords(seedwords,topn = 8,augment_epoch = 3 , del_epoch = 10):
    wordlist = {}
    new_words = []
    for word in tqdm(seedwords):
        aug_word,new_word = augment_word_loop([word],augment_epoch,topn = topn)  # epoch = 5 ,139s
        wordlist[word] = list(set(aug_word))
        new_words.extend(new_word)
        print('\n The keyword is : %s'%word)
        print('new word is:%s'%new_word)
        if aug_word:
            wordlist[word] = delete_word_loop(wordlist[word],epoch = del_epoch)
    # 变成dataframe
    word_dataframe = dict2dataframe(wordlist)
    return word_dataframe,new_words

 if __name__ == '__main__':
    # 一些gensim用法
    pd.Series(model.most_similar('网络营销',topn = 20))

    # 计算两个词的相似度/相关程度
     model.similarity(u"不错", u"好")

    # 寻找对应关系，词类比
    model.most_similar([u'微信', u'营销'], ['网络'], topn=3)

    # 寻找不合群的词
    model.doesnt_match(u"书 书籍 教材 很".split())

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec.load('word2vec_wx')

    # run
    seedwords = ('原型设计','交互设计','产品','策略','经营','市场','社交','电商','文案','排版','社群','媒体','微信','公众号','公关','推广','策划','营销','成本','渠道','口碑','用户画像','体验','数据')
    word_dataframe,new_words = ExpandWords(seedwords,topn = 30 ,augment_epoch = 2, del_epoch = 10)
    # word_dataframe[word_dataframe['words']=='媒体']

    # save
    word_dataframe.to_csv('word_dataframe.csv',encoding = 'utf-8',index =False)
	
	