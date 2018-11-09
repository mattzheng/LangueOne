## 工具篇
主要是分析时候用到的一个小工具。

### 20181109更新
tfidf.py,更新了一个手动写的计算tfidf的小函数：

    tfidf(jieba_content,min_count = 5)

其中：
jieba_content为list型,min_count最小保留词频数

示范案例：

    tfidf([[1,2,3,4,5,6],[3,5,2,8,9]],min_count = 0)
