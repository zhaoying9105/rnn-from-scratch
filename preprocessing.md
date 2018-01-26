# preprocessing
数据预处理

## getSentenceData
功能是获取一个句子向量

1. 读取句子，然后给每个句子增加前后缀，表示矩阵的开始和结束
```python
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))
```
  
2. 用nltk将句子切成词语组成的列表，同时取消长度 <= 3 的句子
```pyhton
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Filter the sentences having few words (including SENTENCE_START and SENTENCE_END)
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))
```

3. 构建词汇字典

使得我们可以根据索引获得词语，也可以根据词语找到索引
```python
# vocab的元素是元组，（词汇，词汇出现的频率），按照频率从高到低排列
vocab = word_freq.most_common(vocabulary_size-1)
# 得到词汇列表
index_to_word = [x[0] for x in vocab]
# 加上  unknown_token
index_to_word.append(unknown_token)
# 组成（词汇，索引）元组序列
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
```

4. 把tokenized_sentences  从词语组成的列表变成索引组成的列表
```python
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
```

5. 构建数据集

```python
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
```
注意这里 X_train 和 y_train的关系
