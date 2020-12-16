#!/usr/bin/python
# -*- coding: utf-8 -*-

from fastHan import FastHan
from opencc import OpenCC

similar_words_path = './data/similar_words.txt'

def load_similar_words(n = -1, path = similar_words_path):
    '''
    加载'说'的近义词
    n: 取与'说'最相似的前n个词作为近义词
    path: 相似词存储路径，文件中的词按照与'说'的相似度排序
    '''
    f = open(path)
    data = f.read()
    data = data.split()
    f.close()
    if n>0 and n <= len(data):
        return data[0:n]
    else:
        return data

class dep_node():
    '''
    依存关系树节点
    '''
    def __init__(self):
        self.parent = None
        self.children = []
        self.ind = None #在原句子中的序号
        self.word = None #词语
        self.relation = None #关系
        self.pos = None #词性

def sen2tree(dep_result):
    '''
    将依存分析的结果转化为树
    dep_result: FastHan模型输出的依存分析结果
    '''
    result = []
    for i in range(len(dep_result[0])+1):
        a = dep_node()
        result.append(a)
    result[0].pos = 'root'
    result[0].ind = 0
    for i in range(len(dep_result[0])):
        item = dep_result[0][i]
        result[i+1].word = item.word #item.word 词语
        result[i+1].parent = result[item.head]  #item.head 父节点的编号
        result[item.head].children.append(result[i+1])
        result[i+1].ind = i+1
        result[i+1].relation = item.head_label #item.head_label 与父节点的关系
        result[i+1].pos = item.pos #item.pos 词性
    return result

def to_simplified(s):
    '''
    繁体->简体
    '''
    cc = OpenCC('t2s')
    return cc.convert(s)

def get_word_ind(dep_tree, node_ind):
    '''
    提取人物或观点字符串
    dep_tree:依存关系树
    node_ind:父节点序号
    return: 人物或观点字符串
    '''
    queue = [node_ind] #待加入节点，列表中全是节点序号ind
    answer_queue = []
    while queue:
        if dep_tree[queue[0]].children:
            for node in dep_tree[queue[0]].children:
                queue.append(node.ind)
        answer_queue.append(queue.pop(0))
    answer_queue = sorted(answer_queue)
    return answer_queue

def extrat_info(dep_tree, node, ner_ind):
    '''
    从依存关系树中提取指定'说'节点下面的人物和观点
    dep_tree: 依存关系树
    node: '说'的近义词的节点
    ner_ind:句子中word为命名实体的位置序列列表,对于speaker,若父节点为代词(pos='PN')，则指代消解
    return: {人物：观点}
    '''
    speaker = []
    point = []
    for child in node.children:
        if child.relation == 'nsubj':
            speaker.append(child.ind)
        else:
            point.append(child.ind)
    if speaker == None:  #如果没有nsubj则选取第一个为主语
        speaker = point.pop(0)
    speaker_str = ind2str(dep_tree, speaker, ner_ind)
    point_str = ind2str(dep_tree, point)
    return (speaker_str,point_str)

def ind2str(dep_tree, ind_list, ner_ind = None):
    '''
    根据序列号生成相应的字符串
    dep_tree: 依存关系树
    ind_list: '说'下方的第一级的单词序号表
    ner_ind: 命名实体序号表
    '''
    answer_ind = [] #要输出的文字对应的序号列表
    for ind in ind_list:
        answer_ind.extend(get_word_ind(dep_tree, ind))
    answer_ind = sorted(answer_ind)
    if ner_ind:
        for ind in ind_list:
            array_ahead = []
            if dep_tree[ind].pos == 'PN':
                array_ahead = [i for i in ner_ind if i<ind] #ner_ind中所有在node_ind之前的元素
            if array_ahead:  #代词指向最近的一个ner
                replace_ind = answer_ind.index(ind) 
                answer_ind[replace_ind] = array_ahead[-1]
    answer = ''
    for ind in answer_ind:
        answer = answer + dep_tree[ind].word
    return answer


def get_info(sentence, n = -1):
    '''
    人物与观点的信息提取
    sentence: 待提取观点的句子
    answer: 字典，key为人物，value为任务相应的观点
    '''
    s_simp = to_simplified(sentence) #繁转简
    sim_words = load_similar_words(n) #载入'说'的近义词
    
    #依存关系分析
    model=FastHan()
    dep_answer=model(s_simp,target="Parsing")
    dep_tree = sen2tree(dep_answer) 
    
    #命名实体识别
    ner=model(s_simp,target="NER")
    ner_word = [ner[0][i].word for i in range(len(ner[0]))]
    ner_ind = [i for i in range(len(dep_tree)) if dep_tree[i].word in ner_word]
    
    #信息提取
    answer = {}
    for node in dep_tree:
        if node.pos == 'VV' and node.word in sim_words:
            speaker, point = extrat_info(dep_tree, node, ner_ind)
            answer[speaker] = point
    
    return answer


def split_first(sentence, split = '。'):
    '''
    先分割句子再提取信息。比直接提取信息效果通常更好。
    '''
    sentences = sentence.split(split)
    result = {}
    for s in sentences:
        result.update(get_info(s))
    return result


if __name__ == '__main__':
    sentence = input('输入需要分析的句子：\n')
    #result = get_info(sentence)
    result = split_first(sentence)
    for key, value in result.items():
        print(f'speaker:{key}, point:{value}\n')