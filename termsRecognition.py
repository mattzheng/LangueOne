# -*- coding: utf-8 -*-


import sys 
#reload(sys)
#sys.setdefaultencoding('utf8')
import warnings
warnings.filterwarnings("ignore")
import regex as re
import numpy as np
import math
import pandas as pd
import jieba
from tqdm import tqdm


class termsRecognition(object):
	def __init__(self, content='',  topK=-1, tfreq=10, tDOA=0, tDOF=0, is_jieba= False,mode = [1]):
		'''
		参数：
			content: 待成词的文本
			maxlen: 词的最大长度
			topK: 返回的词数量
			tfreq: 频数阈值
			tDOA: 聚合度阈值
			tDOF: 自由度阈值
			mode：词语生成模式，一共四种模式，其中第二种模式比较好,一定要写成[1]
			diction:字典，第一批Jieba分词之后的内容
			idf_diction:在第一批字典之后，又生成一批tuple words 的idf，计算方式是，两个词语的平均
			punct:标点符号，Jieba分词之后删除

		步骤：
			jieba_tuples_generator，  利用Jieba分词，并去除标点符号，去除清除''(写入self.jieba_content)，利用wordsGenerator函数生成词语对（四种模式）(写入self.tuple_content)
			word_get_frequency_idf，计算freq 以及贴idf,同时生成'left'/right框，把词语对（self.tuple_content)）写入result(★ 主要写入部分)
			get_doa：只输入result,计算数据的doa，直接更新result中的['doa']
			word_get_dof:只输入result,计算数据的dof，直接更新result中的['dof'],左熵的文字,右熵的文字
			get_score,只输入result,更新result中的['scores']

		可用的函数:
			get_idf,文档的IDF计算
			wordsGenerator,生成词语对
			get_entropy计算左、右熵值，填充result

		'''
		self.content = content
		self.jieba_content = ''
		self.tuple_content = []  # 多项式文本
		self.topK = topK
		self.tfreq = tfreq
		self.tDOA = tDOA
		self.tDOF = tDOF
		self.mode = mode
		self.is_jieba = is_jieba
		self.diction = pd.DataFrame()
		self.idf_diction = {}
		self.punct = ''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''
		self.result = {}


	def jieba_tuples_generator(self):
		print('Cut the words ...')
		if self.is_jieba:
			self.diction = self.get_idf(self.content)  # 只有多行的才能计算IDF
			if isinstance (self.content,tuple):
				self.jieba_content = sum(self.content,())
			elif isinstance (self.content,list):
				self.jieba_content = sum(self.content,[])
		else:
			if isinstance (self.content,str):  
				self.jieba_content = list(jieba.cut(self.content))
				self.jieba_content = list(map( lambda s: ''.join(filter(lambda x: x not in self.punct, s))  , self.jieba_content  ))
				self.jieba_content = list(pd.Series(self.jieba_content) [  [i !='' for i in self.jieba_content]  ].values)   # 清除''
			else:
				self.jieba_content = [list(jieba.cut(i)) for i in self.content]
				self.diction = self.get_idf(self.jieba_content)  # 只有多行的才能计算IDF
				self.jieba_content = list(map( lambda s: ''.join(filter(lambda x: x not in self.punct, s))  , sum(self.jieba_content,[]) ))
				self.jieba_content = list(pd.Series(self.jieba_content) [  [i !='' for i in self.jieba_content]  ].values)  # 清除''
		print('Calculate tuple words ...')
		for t_mode in self.mode:
			self.tuple_content.extend( self.wordsGenerator( self.jieba_content,t_mode)  )


	def word_get_frequency_idf(self):
        # 作用：计算freq 以及贴idf,同时生成'left'/right框，直接写入result
        # 词粒度
		print ('Calculate Frequency for each possible words')
		reg = [i[1] for i in self.tuple_content]              #
		for r in reg:
			if r in self.result:
				self.result[r]['freq'] += 1
				self.result[r]['idf'] = self.idf_diction[r]
			else:
				self.result[r] = {'left':[], 'right':[]}
				self.result[r]['freq'] = 1
				self.result[r]['idf']  = self.idf_diction[r]


	def docs(self,w, D):
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


	def get_idf(self,jieba_content):
		'''
		作用：生成、计算idf
		参考：http://www.voidcn.com/article/p-mhebqvic-qq.html
		输入：分行的文本内容;
		输出：dataframe,分别有:words /  tf / idf  三列
		'''
		print('Calculate IDF ...')
		try:
			diction = pd.DataFrame( list(set(sum(jieba_content,[])) ) , columns = ['words'])
		except:
			diction = pd.DataFrame( list(set(sum(jieba_content,())) ) , columns = ['words'])
		diction['tf'] =  list(map(lambda x : self.docs(x,jieba_content)  , diction['words'] ))
		n = len(jieba_content)
		diction['idf'] = [math.log(i) for i in   n*1.0 / (diction['tf'] + 1 ) ]
		return diction

	def get_doa(self, base=2):
		'''
			pa = 当前词/ 文档数量
			pl = 前词
			pr = 后词
			P(S)/(P(sl)×P(sr))的最小值，取对数之后即可作为聚合度的衡量
		'''
		# 使用信息熵衡量每个词语的聚合度
		print ('Calculate DOA for each possible words')
		for key, value in tqdm(self.result.items()):
			if len(key) == 1:
				self.result[key]['doa'] = 0
				continue
			doa = 99999
			for x in range(1, len(key)):
				try:
					pa = float(self.result[key]['freq']) / len(self.content)
					pl = float(self.result[key[:x]]['freq']) / len(self.content)
					pr = float(self.result[key[x:]]['freq']) / len(self.content)
					td = math.log(pa / (pl * pr), base)
					if td < doa:
						doa = td
				except:
					pass
				else:
					pass
				finally:
					pass
			self.result[key]['doa'] = doa
    

	def wordsGenerator(self,base_text , mode = 1):
		'''
		输入 ：jieba.cut分词
		输出 ：List,[('f', '司法', '解'),('司', '法解', '释')]
		譬如：相对来说,1比较适合
        		'这个函数真的比较特殊'-> '这个,函数,真的,比较,特殊'
        		
        		mode == 1:
        		词,词+词，词   -->'这个,函数真的,比较' + '函数,真的比较,特殊'
        			
        		mode == 2:（效果最佳）
        		词，词+词+词，词   -->'这个,函数真的比较,特殊'  
        			
        		mode == 3:
        		词，字+词，词  /  词，词+ 字，词  -->'这个,个函数,真的'  ,  '这个,函数真,真的'
        
        		mode == 22(fail): 
        		字+字，词+词，字+字   -->'这个,函数真的,比较' + '函数,真的比较,特殊'
			
                  mode == 4:
                  词，词+词+词+词，词   -->'这个,函数真的比较特殊'
		'''
		base_word = []
		
		if mode == 1 :
			for i in  tqdm(range(1,len(base_text)-3)):
				base_word.append((base_text[i-1] , base_text[i] + ' ' + base_text[i+1] , base_text[i+2]))
				# 计算idf
				#print('word generator mode 1 ,Round at %s ...' %i)
				self.idf_diction[base_text[i] + ' ' + base_text[i+1]] = np.hstack((self.diction[ (self.diction['words']==base_text[i])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+1])  ]['idf'].values)).mean()

		if mode == 2 :
			for i in  tqdm(range(1,len(base_text)-4)):
				base_word.append((base_text[i-1] , base_text[i] + ' ' + base_text[i+1] + ' ' +  base_text[i+2] , base_text[i+3]))
    				# 计算idf
				#print('word generator mode 2 ,Round %s ... ' %i)
				self.idf_diction[ base_text[i] + ' ' + base_text[i+1] + ' ' +  base_text[i+2]] = np.hstack((self.diction[ (self.diction['words']==base_text[i])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+1])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+2])  ]['idf'].values)).mean()

		if mode == 3:
			for i in  tqdm(range(1,len(base_text)-3)):
				base_word.append(( base_text[i-1] , base_text[i-1][-1] + ' ' +  base_text[i] , base_text[i+1] ))
				base_word.append(( base_text[i-1] ,  base_text[i] +  ' ' +  base_text[i+1][0], base_text[i+1] ))
    				# 计算idf
				#print('word generator mode 3 ,Round at %s ...' %i)
				self.idf_diction[base_text[i-1][-1] +  ' ' +  base_text[i]] = np.hstack((self.diction[ (self.diction['words']==base_text[i])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i-1])  ]['idf'].values)).mean()
				self.idf_diction[base_text[i] +  ' ' +  base_text[i+1][0]] = np.hstack((self.diction[ (self.diction['words']==base_text[i])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+1])  ]['idf'].values)).mean()

		if mode == 4 :
			for i in  tqdm(range(1,len(base_text)-5)):
				base_word.append((base_text[i-1] , base_text[i] + ' ' +   base_text[i+1] + ' ' +   base_text[i+2] + ' ' +   base_text[i+3], base_text[i+4]))
    				# 计算idf
				#print('word generator mode 4 ,Round at %s ...' %i)
				self.idf_diction[base_text[i] + ' ' +   base_text[i+1] + ' ' +   base_text[i+2] + ' ' +   base_text[i+3]] = \
					np.hstack((self.diction[ (self.diction['words']==base_text[i])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+1])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+2])  ]['idf'].values,\
					self.diction[ (self.diction['words']==base_text[i+3])  ]['idf'].values)).mean()

		return base_word

	def word_get_dof(self):
		'''
		根据tuple_content的词语对，填充result中的['left'] / ['right']两列
		每个词，根据填充好的left/ right的词语，通过get_entropy计算左、右熵值，填充result
		'''
		# 使用信息熵衡量每个词语的自由度
		print('Calculate DOF for each possible words')
		reg = self.tuple_content
		for r in reg:
			self.result[r[1]]['left'].append(r[0])
			self.result[r[1]]['right'].append(r[2])
		for key, value in tqdm(self.result.items()):
			left = self.get_entropy(self.delete_boxex(value['left']))
			right = self.get_entropy(self.delete_boxex(value['right']))
			if left < right:
				self.result[key]['dof'] = left
			else:
				self.result[key]['dof'] = right

	def delete_boxex(self ,strings):
		[i != [] for i in strings]
		return list(pd.Series(strings) [ [i != [] for i in strings ]  ] .values)

    
	def get_entropy(self, data, base=2):
		'''
			根据tuple_content的词语对，填充result中的['left'] / ['right']两列
			每个词，根据填充好的left/ right的词语，通过get_entropy计算左、右熵值，填充result
		'''
		tmp = {}
		for item in data:
			if not item in tmp:
				tmp[item] = 1.0
			else:
				tmp[item] += 1.0
		for key, value in tmp.items():
			tmp[key] /= float(len(data))
		result = 0.0
		for key, value in tmp.items():
			result += value * math.log(value, base)
		if result < 0:
			result = -result
		return result

	def get_score(self):
		# 将频数、聚合度、自由度归一化，并计算总得分
		print ('Calculate Score for each possible words')
		for key, value in tqdm(self.result.items()):
			if value['freq'] <= self.tfreq or value['doa'] <= self.tDOA or value['dof'] <= self.tDOF:
				self.result[key]['score'] = 0
			else:	
				self.result[key]['score'] = value['freq'] * value['doa'] * value['dof'] * value['idf']


	def generate_word(self):
		'''
		步骤：
			jieba_tuples_generator，  利用Jieba分词，并去除标点符号，去除清除''(写入self.jieba_content)，利用wordsGenerator函数生成词语对（四种模式）(写入self.tuple_content)
			word_get_frequency_idf，计算freq 以及贴idf,同时生成'left'/right框，把词语对（self.tuple_content)）写入result(★ 主要写入部分)
			get_doa：只输入result,计算数据的doa，直接更新result中的['doa']
			word_get_dof:只输入result,计算数据的dof，直接更新result中的['dof'],左熵的文字,右熵的文字
			get_score,只输入result,更新result中的['scores']
		'''
		self.jieba_tuples_generator()
		self.word_get_frequency_idf()
		self.get_doa()
		self.word_get_dof()
		self.get_score()
		result = sorted(self.result.items(), key=lambda d:d[1]['score'], reverse=True)
		if self.topK == -1:
			return result
		else:
			return result[:self.topK]

	def get_result(self):
		result = []
		for key,values in self.result.items():
			result.append([key,values['dof'],values['doa'], values['freq'] , values['score'] ,values['idf'] ])
		return pd.DataFrame(result,columns = ['key','dof','doa','freq','score','idf'])
  
	def part_found(self):
		'''
		部分发现:
		只计算词频freq以及idf值
		计算之后，生成dataframe表格
		'''
		print('local found ... ')
		generator.jieba_tuples_generator()
		generator.word_get_frequency_idf()
		result_dataframe = pd.DataFrame(generator.result).T
		result_dataframe['words'] = result_dataframe.index
		result_dataframe.reset_index(inplace=True,drop=True)
		return result_dataframe


# --------------------------- 词粒度 评估 --------------------------- 
if __name__ == '__main__':
    data = pd.read_csv('toutiao_data.csv',encoding = 'utf-8')
    
    # ----------- 未分词 ----------- 
    generator = termsRecognition(content = data['new_title'][:10000] ,is_jieba=False, topK = 20 , mode = [1])   # 文字版
    # 全部发现
    result_dict = generator.generate_word()
    result_dataframe = generator.get_result()
    # 部分发现
    result_dataframe = generator.part_found()
    
    
    # ----------- 已分词 ----------- 
    def not_nan(obj):
        return obj == obj
    
    keywords = []
    for word in tqdm(data.new_keyword):
        if not_nan(word):
            keywords.append(word.split(','))
    
    
    generator = termsRecognition(content = keywords[:1000] , is_jieba=True , topK = 20,mode = [1]) #图像版
    result_dataframe = generator.part_found()
