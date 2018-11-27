## 工具篇
主要是分析时候用到的一个小工具。

### 20181109更新
tfidf.py,更新了一个手动写的计算tfidf的小函数：

    tfidf(jieba_content,min_count = 5)

其中：
jieba_content为list型,min_count最小保留词频数

示范案例：

    tfidf([[1,2,3,4,5,6],[3,5,2,8,9]],min_count = 0)

### 20181127更新
监督式词重要性判定，核心参考于kaggle恶意评价分类的比赛之中：

https://www.kaggle.com/tks0123456789/word-phrase-importance

思路：

先得到词条/单词的tfidf矩阵，然后通过向量去监督训练（使用的是LR模型），然后根据模型的回归系数作为权重。

启迪:

词条的tfidf代表词语对于整个文本的重要性； 这种方式可以得到词条对于分类的重要性，是有监督的一种方式

