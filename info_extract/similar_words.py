#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import reduce

import pandas as pd
from gensim.models import KeyedVectors


def get_similar_words(filename, word = ['说'], n1 = 10, n2 = 500, model_path = './data/', ):
    '''
    载入预训练词向量，获取word的近义词
    预训练词向量：https://github.com/Embedding/Chinese-Word-Vectors
    filename: 文件名
    word: 获取近义词的目标
    n1, n2: 先获取word的前n1个近义词，之后再以word和这n1个近义词作为目标，获取最相似的前n2个近义词
    model_path: 词向量文件夹
    return: word + n1 + n2 对应的词
    '''
    wv = KeyedVectors.load_word2vec_format(model_path+filename,binary=False, encoding="utf8",  unicode_errors='ignore')
    similar_words = wv.most_similar(positive = word, topn = n1) #前n1个
    df = pd.DataFrame(similar_words)
    
    df0 = pd.DataFrame({0:word, 1:[1]*len(word)}) 
    df1 = df.iloc[0:n1]
    df01 = pd.concat([df0, df1], ignore_index = True) #word+n1

    similar_words = wv.most_similar(positive = df01[0], topn = n2)
    df3 = pd.DataFrame(similar_words)

    return pd.concat([df01, df3], ignore_index = True)

def gen_sim_words(filenames, save = True, model_path = './data/'):
    '''
    从多个预训练的词向量中获取近义词并合并
    filenames: 预训练词向量文件列表
    save: 是否向硬盘保存中间结果
    model_path: 词向量文件存储路径
    return：综合起来的近义词列表
    '''
    sim_words_list = [] #临时存储每种词向量中找到的近义词，之后合并
    for filename in filenames:
        df = get_similar_words(filename)
        if save == True:
            df.to_csv(model_path+filename+'.csv', index = False)
        sim_words = df[0]
        sim_words_list.append(sim_words)
        print(f'{filename}完成\n')
    result = reduce(lambda x,y: set(x) | set(y), sim_words_list)
    return result


filenames = ['sgns.renmin.bigram', 'sgns.sogou.bigram', 'sgns.wiki.bigram']
sim_words = list(gen_sim_words(filenames))
f = open('./data/similar_words.txt', mode = 'w')
f.write(' '.join(sim_words))
f.close()