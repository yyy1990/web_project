#!/usr/bin/python
# -*- coding: utf-8 -*-

import jieba
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

import sentiment.convert_input as convert_input

#预处理，数据来源：https://github.com/CLUEbenchmark/CLUEDatasetSearch


def data_preprocess(vector_paths, data_path, save_path):
    '''
    对已经转换为DataFrame形式的数据预处理
    vector_paths: 预训练向量路径，依次为人民日报、搜狗新闻、维基百科语料库预训练词向量
    weibo_path: 原始数据存放路径
    save_path: 处理完成后的存储路径
    '''
    weibo_df = pd.read_csv(data_path, encoding = 'utf-8')
    
    #用人民日报预训练向量处理
    vector_rm = KeyedVectors.load_word2vec_format(vector_paths[0],binary=False, encoding="utf8",  unicode_errors='ignore')
    weibo_df.rename(columns = {'review': 'microblog'}, inplace = True)

    cut_list = [] #已完成分词的句子序列
    l_list = [] #分词长度序列
    l_rm_list = [] #用人民日报预训练向量处理的长度
    for i, row in weibo_df.iterrows():
        temp_cut = list(jieba.cut(row['microblog']))
        cut_list.append(' '.join(jieba.cut(row['microblog'])))
        l_list.append(len(temp_cut))
        temp_vec = convert_input.cut_sentence(temp_cut, vector_rm)  
        l_rm_list.append(len(temp_vec))

    weibo_df['cut'] = cut_list
    weibo_df['len'] = l_list
    weibo_df['len_rm'] = l_rm_list

    #用搜狗新闻预训练向量处理
    vector_sg = KeyedVectors.load_word2vec_format(vector_paths[1],binary=False, encoding="utf8",  unicode_errors='ignore')
    weibo_df = wb_pre_vec(vector_sg, weibo_df, 'len_sg')

    #用维基百科预训练向量处理
    vector_wk = KeyedVectors.load_word2vec_format(vector_paths[2],binary=False, encoding="utf8",  unicode_errors='ignore')
    weibo_df = wb_pre_vec(vector_wk, weibo_df, 'len_wk')

    weibo_df.to_csv(save_path, encoding = 'utf-8', index = False)
    return weibo_df

def wb_pre_vec(vector, df, col_name):
    '''
    用不同的预训练向量处理
    vector: 预训练向量
    df: 原始数据DataFrame
    col_name: 新增列名称
    '''
    l_list = []
    for i, row in df.iterrows():
        temp_cut = row['cut'].split()
        temp_vec = convert_input.cut_sentence(temp_cut, vector)
        l_list.append(len(temp_vec))
    df[col_name] = l_list
    return df


def weibo100k_preprocess():
    '''
    对微博100k数据的预处理
    '''
    renmin_path = './data/sgns.renmin.bigram'
    sogou_path = './data/sgns.sogou.bigram'
    wiki_path = './data/sgns.wiki.bigram'
    vector_paths = [renmin_path, sogou_path, wiki_path]
    weibo_path = './data/weibo_100k/weibo_senti_100k.csv'
    save_path = './data/weibo_cut.csv'
    weibo_df = data_preprocess(vector_paths, weibo_path,save_path)


def nlpcc_preprocess():
    '''
    对NLPCC2014 task2 数据的预处理
    '''
    renmin_path = './data/sgns.renmin.bigram'
    sogou_path = './data/sgns.sogou.bigram'
    wiki_path = './data/sgns.wiki.bigram'
    vector_paths = [renmin_path, sogou_path, wiki_path]
    nlpcc_path = './data/NLPCC2014 Task2/sample_combined.csv'
    save_path = './data/nlpcc_cut.csv'
    nlpcc_df = data_preprocess(vector_paths, nlpcc_path,save_path)


def split_df(path, ratio):
    '''
    将指定路径的dataframe按照
    path: 文件路径
    ratio: 训练数据所占比例
    '''
    df = pd.read_csv(path, encoding = 'utf-8')
    unique_labels = df['label'].unique()
    answer = {}
    for label in unique_labels: #对于所有标签分别抽取，确保后类别比例不变
        temp_df = df[df['label'] == label]
        index = np.array(temp_df.index)
        np.random.shuffle(index)
        l = len(temp_df)
        l_train = np.round(l*ratio).astype(np.int32)
        index_train= index[0:l_train]  #标签为label的训练集数据index
        index_test = index[l_train::]
        answer[label] = {'train': temp_df.loc[index_train], 'test': temp_df.loc[index_test]}
    return answer


def gen_train_test_set(ratio, num, weibo_path='./data/weibo_cut.csv', nlpcc_path='./data/nlpcc_cut.csv'):
    '''
    生成训练集与测试集
    ratio: 训练集所占比例
    num:生成几组
    weibo_path: 微博数据路径
    nlpcc_path: NLPCC2014 task2 数据路径
    '''
    for i in range(num):
        weibo_split = split_df(weibo_path, ratio)
        nlpcc_split = split_df(nlpcc_path, ratio)
        train_df = [weibo_split[0]['train'], weibo_split[1]['train'], nlpcc_split[0]['train'], nlpcc_split[1]['train']]
        test_df = [weibo_split[0]['test'], weibo_split[1]['test'], nlpcc_split[0]['test'], nlpcc_split[1]['test']]
        train_df = pd.concat(train_df, axis = 0)
        test_df = pd.concat(test_df, axis = 0)
        train_df.to_csv(f'./data/train_{i}.csv', encoding = 'utf-8', index = False)
        test_df.to_csv(f'./data/test_{i}.csv', encoding = 'utf-8', index = False)
    return True

def gen_train_test_set_weibo(ratio, num, weibo_path='./data/weibo_cut.csv'):
    '''
    生成训练集与测试集
    ratio: 训练集所占比例
    num:生成几组
    weibo_path: 微博数据路径
    '''
    for i in range(num):
        weibo_split = split_df(weibo_path, ratio)
        train_df = [weibo_split[0]['train'], weibo_split[1]['train']]
        test_df = [weibo_split[0]['test'], weibo_split[1]['test']]
        train_df = pd.concat(train_df, axis = 0)
        test_df = pd.concat(test_df, axis = 0)
        train_df.to_csv(f'./data/train_{i}.csv', encoding = 'utf-8', index = False)
        test_df.to_csv(f'./data/test_{i}.csv', encoding = 'utf-8', index = False)
    return True

if __name__ == '__main__':
    weibo100k_preprocess()
    nlpcc_preprocess()
    gen_train_test_set(0.8, 10)