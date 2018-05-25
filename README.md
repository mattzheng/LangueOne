# LangueOne
练习题︱基于今日头条开源数据的词共现、新热词发现、短语发现

最近笔者在做文本挖掘项目时候，写了一些小算法，不过写的比较重，没有进行效率优化，针对大数据集不是特别好用，不过在小数据集、不在意性能的情况下还是可以用用的。

**本次练习题中可以实现的功能大致有三个：**

 - 短语发现
 - 新词发现
 - 词共现

**短语发现、新词发现跟词共现有些许区别：**
[‘举’，'个'，‘例子’，‘来说’]

 - 短语发现、新词发现，是词-词连续共现的频率，窗口范围为1，也就是：‘举’，‘例子’；'个'，‘例子’；‘例子’，‘来说’，探究挨得很近的词之间的关系
 - 词共现是词-词离散出现，词共现包括了上面的内容，探究：‘举’，‘来说’，不用挨着的词出现的次数


----------


## 一、数据集介绍
练习数据来源：[今日头条中文新闻（文本）分类数据集](https://github.com/fateleak/toutiao-text-classfication-dataset)

今日头条是最近开源的数据集，38w，其中的数据格式为：


```
6552391948794069256_!_106_!_news_house_!_新手买房，去售楼部该如何咨询？_!_
6552263884172952072_!_106_!_news_house_!_南京90后这么有钱吗？南京百分之四五十都是小杆子买了_!_公积金,江宁,麒麟镇,南京90后,大数据
6552313685874835726_!_106_!_news_house_!_涨价之前买房的人，现在是什么心情？_!_
6552447172724392456_!_106_!_news_house_!_这种凸阳台房子万万不要买，若不是售楼闺蜜说，我家就吃大亏_!_凸阳台,售楼,买房
```

每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词


----------


## 二、短语发现、新词发现算法介绍

### 2.1  理论介绍

短语发现、新词发现，内容与算法基础源于该博客：[基于凝聚度和自由度的非监督词库生成](http://zhanghonglun.cn/blog/project/%E5%9F%BA%E4%BA%8E%E5%87%9D%E8%81%9A%E5%BA%A6%E5%92%8C%E8%87%AA%E7%94%B1%E5%BA%A6%E7%9A%84%E9%9D%9E%E7%9B%91%E7%9D%A3%E8%AF%8D%E5%BA%93%E7%94%9F%E6%88%90/)


评估词之间的几个指标，出了频率还有：


 - 凝聚度：

   

>  该词语为S，首先计算该词语出现的概率P(S)，然后尝试S的所有可能的二切分，即分为左半部分sl和右半部分sr并计算P(sl)和P(sr)，
>     例如双汉字词语存在一种二切分、三汉字词语存在两种二切分。接下来计算所有二切分方案中，P(S)/(P(sl)×P(sr))的最小值，取对数之后即可作为聚合度的衡量。
>     以双汉字词语为例，可以想象到，如果该词语的聚合度很低，说明其第一个字和第二个字的相关性很弱，甚至是不相关的，那么P(S)和P(sl)×P(sr)将处于同一个数量级。
>     相反，如果该词语的聚合度很高，“齐天”、“大圣”和“齐天大圣”三者的概率都很接近，因此P(S)/(P(sl)×P(sr))将是一个远大于1的数值。

 - 自由度：

>     用熵来衡量一个词语的自由度。假设一个词语一共出现了N次，其左边共出现过n个汉字，每个汉字依次出现N1，N2，……，Nn次，则满足N = N1 + N2 + …… + Nn，因此可以计算该词语左边各个汉字出现的概率，
>     并根据熵公式计算左邻熵。熵越小则自由度越低，例如“天大圣”的左邻熵接近于0，因为“齐”字的概率几乎为1；熵越大则自由度越高，表示用词搭配越混乱、越自由、越多样。
>     因为“天大圣”的左邻熵很小，而右邻熵则相对较大，因此我们将一个词语左邻熵和右邻熵中较小者作为最终的自由度。

 - IDF:

>     逆文档词频


### 2.2 主函数参数

算法的参数描述：

```
class termsRecognition(object):
	def __init__(self, content='',  topK=-1, tfreq=10, tDOA=0, tDOF=0, is_jieba= False,mode = [1]):
```

其中的参数：：


 - content: 待成词的文本
 - maxlen: 词的最大长度

 - topK: 返回的词数量
 - tfreq: 频数阈值
 - tDOA: 聚合度阈值
 - tDOF: 自由度阈值
 - mode：词语生成模式，一共四种模式，其中第二种模式比较好,一定要写成[1]
 - diction:字典，第一批Jieba分词之后的内容
 - idf_diction:在第一批字典之后，又生成一批tuple words 的idf，计算方式是，两个词语的平均
 - punct:标点符号，Jieba分词之后删除


算法步骤：

 - jieba_tuples_generator， 
   利用Jieba分词，并去除标点符号，去除清除''(写入self.jieba_content)，利用wordsGenerator函数生成词语对（四种模式）(写入self.tuple_content)
 - word_get_frequency_idf，计算freq
   以及贴idf,同时生成'left'/right框，把词语对（self.tuple_content)）写入result(★ 主要写入部分)
 - get_doa：只输入result,计算数据的doa，直接更新result中的['doa']
 - word_get_dof:只输入result,计算数据的dof，直接更新result中的['dof'],左熵的文字,右熵的文字
 - get_score,只输入result,更新result中的['scores']


可用的函数:

 - get_idf,文档的IDF计算
 - wordsGenerator,生成词语对
 - get_entropy计算左、右熵值，填充result

----------


## 三、词共现算法介绍

就是计算词语共同出现的概率，一般用在构建词条网络的时候用得到，之前看到这边博客提到他们自己的算法：《[python构建关键词共现矩阵](https://blog.csdn.net/alanconstantinelau/article/details/69258443)》看着好麻烦，于是乎自己简单写了一个，还是那个问题....效率比较低...

之前一般的做法是先生成一个基于词-词矩阵，然后去累计词-词之间的出现次数。

![这里写图片描述](https://images2017.cnblogs.com/blog/958950/201708/958950-20170821101923668-494703642.png)

我这边只是简单利用笛卡尔积:


 - permutations 排列
 - combinations 组合,没有重复
 - combinations_with_replacement 组合,有重复


```
>>> import itertools  
>>> for i in itertools.product('ABCD', repeat = 2):  
...     print i,  
...   
('A', 'A') ('A', 'B') ('A', 'C') ('A', 'D') ('B', 'A') ('B', 'B') ('B', 'C') ('B', 'D') ('C', 'A') ('C', 'B') ('C', 'C') ('C', 'D') ('D', 'A') ('D', 'B') ('D', 'C') ('D', 'D')  
>>> for i in itertools.permutations('ABCD', 2):  
...     print i,  
...   
('A', 'B') ('A', 'C') ('A', 'D') ('B', 'A') ('B', 'C') ('B', 'D') ('C', 'A') ('C', 'B') ('C', 'D') ('D', 'A') ('D', 'B') ('D', 'C')  
>>> for i in itertools.combinations('ABCD', 2):  
...     print i,  
...   
('A', 'B') ('A', 'C') ('A', 'D') ('B', 'C') ('B', 'D') ('C', 'D')  
>>> for i in itertools.combinations_with_replacement('ABCD', 2):  
...     print i,  
...   
('A', 'A') ('A', 'B') ('A', 'C') ('A', 'D') ('B', 'B') ('B', 'C') ('B', 'D') ('C', 'C') ('C', 'D') ('D', 'D') 
```




----------


## 四、练习题

**文件夹介绍：**

 - 短语发现、新词发现算法：termsRecognition.py
 - 今日头条数据38w：toutiao_data.csv
 - 二元组算法：tuplewords.py

**先来看看数据长啥样：**


![这里写图片描述](https://img-blog.csdn.net/20180525181140478?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


**废话不多说，直接使用一下：**

### 4.1 短语发现、新词发现模块

该模块可以允许两种内容输入，探究的是词-词之间连续共现，一种数据格式是没有经过分词的、第二种是经过分词的。
其中，算法会提到全部发现以及部分发现两种模式，这两种模式的区别主要在于考察指标的多少。

 - 全部发现会考察：凝聚度、自由度、IDF、词频
 - 部分发现会考察：IDF、词频


##### 4.1.1 没有经过分词的原始语料

在今日头条数据之中就是标题数据了，一般用来新词发现，这边整体运行很慢，就截取前10000个。
```
data = pd.read_csv('toutiao_data.csv',encoding = 'utf-8')

generator = termsRecognition(content = data['new_title'][:10000] ,is_jieba=False, topK = 20 , mode = [1])   # 文字版
# 全部发现
result_dict = generator.generate_word()
result_dataframe = generator.get_result()
# 部分发现
result_dataframe = generator.part_found()
    
```

得到的结论，如图：
![这里写图片描述](https://img-blog.csdn.net/20180525181402544?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这边其实可以在Jieba分词的时候，预先载入一些停用词。这边来看，发现的有：对下联、王者荣耀
新词发现的能力还不够好。


##### 4.1.2 经过分词的原始语料

比较适合用在已经分完词的语料比较适合：[['经过','分词'],['的','原始'],['原始','语料'],...]
当然，探究的是词-词之间的连续共现的情况。此时，我用今日头条的关键词其实不是特别合适，因为关键词之间没有前后逻辑关系在其中。
在此只是简单给观众看一下功能点。


```
data = pd.read_csv('toutiao_data.csv',encoding = 'utf-8')
def not_nan(obj):
    return obj == obj

keywords = []
for word in tqdm(data.new_keyword):
    if not_nan(word):
        keywords.append(word.split(','))

generator = termsRecognition(content = keywords[:1000] , is_jieba=True , topK = 20,mode = [1]) #图像版
# 部分发现
result_dataframe = generator.part_found()
```
得到的结论：

![这里写图片描述](https://img-blog.csdn.net/20180525181606212?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


其中发现了的规律都没啥用，大家看看就行。。

### 4.2 词共现模块

二元组模块跟4.1中，分完词之后的应用有点像，但是这边是离散的，之前的那个考察词-词之间的排列需要有逻辑关系，这边词共现会更加普遍。
该模块较多会应用在基于关键词的SNA社交网络发现之中，给张好看的图：

![这里写图片描述](https://upload-images.jianshu.io/upload_images/1213137-2d6b93d5a9cd90b8.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/673)


**其中，在该模块写入了两种：**

 - 热词统计
 - 词共现统计

```
data = pd.read_csv('toutiao_data.csv',encoding = 'utf-8')


def not_nan(obj):
    return obj == obj

keywords = []
for word in tqdm(data.new_keyword):
    if not_nan(word):
        keywords.append(word.split(','))
        
# 设置停用词
stop_word = ['方法','结论']
tw = TupleWords(stop_word)

# 得到结果
id_pools,tuple_pools = tw.CoOccurrence(keywords[:1000])
    
# 内容变成dataframe
CoOccurrence_data = tw.tansferDataFrame(tuple_pools)
CoOccurrence_data
        
# 热词统计模块
hotwords_dataframe = tw.Hotwords(keywords[:1000])
hotwords_dataframe
```

该模块输入的是keywords，List形：

```
[['保利集团', '马未都', '中国科学技术馆', '博物馆', '新中国'],
 ['林风眠',
  '黄海归来步步云',
  '秋山图',
  '计白当黑',
  '山水画',
  '江山万里图',
  '张大千',
  '巫峡清秋图',
  '活眼',
  '山雨欲来图'],
 ['牡丹', '收藏价值'],
 ['叶浅予', '田世光', '李苦禅', '花鸟画', '中央美术学院']
```

tw.CoOccurrence就是对上面的内容进行解析，得到了：

![这里写图片描述](https://img-blog.csdn.net/20180525182803137?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

发现，快乐大本营-谢娜的组合比较多，詹姆斯-猛龙嘛，看客们懂的，詹皇血克猛龙，哈哈~

热词发现这个很常规：

![这里写图片描述](https://img-blog.csdn.net/20180525183021368?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


----------


## 后续拓展——SNA社交网络发现网络图：
得到了CoOccurrence_data 的表格，有了词共现，就可以画社交网络图啦，有很多好的博客都有这样的介绍，推荐几篇：

#### [基于共现发现人物关系的python实现](https://www.jianshu.com/p/2c8a81112ad4)
![这里写图片描述](https://upload-images.jianshu.io/upload_images/1213137-7624c508141e4f9a.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/690)


#### [python简单实战项目：《冰与火之歌1-5》角色关系图谱构建——人物关系可视化](https://www.jianshu.com/p/db7e3e4f728d)

![这里写图片描述](https://upload-images.jianshu.io/upload_images/5829213-8b9eab8ce443e2bb?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

