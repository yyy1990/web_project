import jieba
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from opencc import OpenCC


def to_simplified(s):
    '''
    繁体->简体
    '''
    cc = OpenCC('t2s')
    return cc.convert(s)

def df_simplified(filename = 'microblog_data.csv', model_path = './data/', save = True):
    '''
    将训练数据中所有句子繁转简
    '''
    df = pd.read_csv(model_path+filename, encoding = 'utf-8')
    df['microblog'] = df['microblog'].apply(to_simplified)
    if save == True:
        df.to_csv(model_path+filename, encoding = 'utf-8', index = False)
    return df

def load_pretrained_vec(filename, model_path = './data/'):
    '''
    载入词向量
    '''
    wv = KeyedVectors.load_word2vec_format(model_path+filename,binary=False, encoding="utf8",  unicode_errors='ignore')
    return wv

def sub_words(word):
    '''
    返回word的所有子词
    '''
    answer = []
    n = len(word)
    for i in range(1, n):
        start = 0
        while start+i <= n:
            answer.append(word[start:(start+i)])
            start = start + 1
    return answer

def oov(word, pretrained_vec):
    '''
    对于未登录词的处理：借鉴fasttext的方法，将word的所有已登陆的子词的词向量平均
    word:未登录词
    pretrained_vec:预训练词向量
    return: 未登录词的词向量
    '''
    sub_word_list = sub_words(word)
    answer = []
    for w in sub_word_list:
        if w in pretrained_vec:
            answer.append(pretrained_vec[w])
    if answer:
        answer = np.vstack(answer)
        answer = np.mean(answer, axis = 0)
        return answer
    else:
        return None  #如果所有的子词都没有词向量，则跳过

def cut_sentence(text, word_vector, _cut = False):
    '''
    将文本转换为向量
    text:待转换句子
    word_vector:预训练词向量
    _cut:是否需要分词
    return: 二维数组，每一行是一个词的词向量
    '''
    if _cut == True:
        cut = list(jieba.cut(text))
    else:
        cut = text
    answer = []
    for word in cut:
        if word in word_vector:
            vec = word_vector[word]
        else:
            vec = oov(word, word_vector)
        if vec is not None:
            answer.append(vec)
    if len(answer)>1:
        answer = np.vstack(answer)
        return answer
    elif len(answer) == 1:
        return answer[0]
    else:  #如果所有的词及其子词都找不到词向量，则返回None
        return None 


#df = df_simplified()
#filename = 'microblog_data.csv'
#model_path = './data/'
#df = pd.read_csv(model_path+filename, encoding = 'utf-8')
#vec_filename = 'sgns.renmin.bigram'
#vec = load_pretrained_vec(vec_filename)
#vec_stack = cut_sentence
