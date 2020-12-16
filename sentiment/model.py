#!/usr/bin/python
# -*- coding: gbk -*-

import jieba
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sentiment.convert_input as convert_input


def set_device():
    '''
    设置训练设备
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


class dataset(Dataset):
    def __init__(self,data_path,vector,max_len,len_name):
        '''
        data_path:训练数据路径
        vector_path:预训练词向量路径
        max_len:一句话最多多少个词向量
        len_name:标记有效长度的列的名称
        '''
        self.vector = vector
        self.max_len = max_len
        self.len_name = len_name
        df = pd.read_csv(data_path, encoding = 'utf-8')
        self.df = df[df[self.len_name]>1]
    
    def __getitem__(self, index):
        text = self.df.loc[index]['cut']
        sen_vec = convert_input.cut_sentence(text, self.vector) #sen_vec.shape=(时间，词向量维度)
        if sen_vec.shape[0] > self.max_len: #句子长度长于max_len，需要截断
            answer = sen_vec[0:self.max_len]
            l = self.max_len
        elif sen_vec.shape[0] < self.max_len: #句子长度小于max_len，补0
            answer = np.zeros((self.max_len, sen_vec.shape[1]))
            answer[0:sen_vec.shape[0],0:sen_vec.shape[1]] = sen_vec
            l = self.df.loc[index][self.len_name]
        else:
            answer = sen_vec
            l = self.df.loc[index][self.len_name]
        return (answer.astype(np.double), l, self.df.loc[index]['label'])
    
    def __len__(self):
        return len(self.df)


class model(torch.nn.Module):
    def __init__(self, vocab_dim, hidden_dim, num_layers = 1, features = 2, bidir = False):
        super(model, self).__init__()
        '''
        定义模型
        vocal_dim: 词向量维度
        hidden_dim: LSTM隐藏单元维度, 最长时间步
        num_layers: LSTM层数
        features: 取hidden的最后几个时间步的出来用于预测
        bi_dir: LSTM 是否双向
        '''
        self.features = features
        self.rnn = torch.nn.LSTM(input_size = vocab_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True, bidirectional = False)
        self.fc = torch.nn.Linear(num_layers*features, 1)
    
    def forward(self, sen_batch, l):
        '''
        sen_batch: sen_batch:句子序列, (batch_size, time, embedding_dim)
        l: 句子长度，(batch_size, 1)
        '''
        #(sen_batch, l, label) = x
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(sen_batch.float(), l, batch_first = True, enforce_sorted = False)
        output, (hidden, cell) = self.rnn(input_packed)
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size]
        # c_n: 同h_n
        hidden = hidden[:,:,-self.features::] #[num_layers, batch_size, self.features]
        hidden = hidden.transpose(0,1) #[batch_size, num_layers, self.features]
        batch_size = hidden.shape[0]
        num_layers = hidden.shape[1]
        hidden = hidden.view(batch_size, num_layers*self.features)

        output = self.fc(hidden)
        
        return output


def train(rnn, iterator, optimizer, criteon, device):
    '''
    训练
    rnn: 模型
    iterator: 训练数据的dataloader
    optimizer: torch.optim.Adam
    criteon: torch.nn.CrossEntropyLoss
    '''
    avg_acc = []
    avg_loss = []
    rnn.train()

    pbar = tqdm(total = len(iterator))
    for i, batch in enumerate(iterator):
        pbar.set_description(f'batch {i}:')
        (sen_batch, l_batch, label_batch) = batch
        
        sen_batch = sen_batch.to(device)
        l_batch = l_batch.to(device)
        label_batch = label_batch.unsqueeze(1).float().to(device) #shape=(batch)->(batch,1)

        pred = rnn(sen_batch, l_batch)
        loss = criteon(pred, label_batch)
        accuracy = acc(pred, label_batch)

        avg_loss.append(loss.data.item())
        avg_acc.append(accuracy.item()) 
        pbar.set_postfix(Loss = loss.data.item(), Accuracy = accuracy.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()
    return (avg_loss, avg_acc)


def acc(preds, y):
    '''
    准确度计算
    preds:预测值, [batch_size, num_labels]
    y:标签值, [batch_size]
    '''
    #preds = torch.argmax(preds, dim = 1)
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    result = correct.sum()/len(correct)
    return result


def evaluate(rnn, iterator, criteon, device):
    #测试集评估
    avg_acc = []
    avg_loss = []
    rnn.eval()
    with torch.no_grad():
        pbar = tqdm(total = len(iterator))
        for i, batch in enumerate(iterator):
            pbar.set_description(f'batch {i}:')
            (sen_batch, l_batch, label_batch) = batch

            sen_batch = sen_batch.to(device)
            l_batch = l_batch.to(device)
            label_batch = label_batch.unsqueeze(1).float().to(device)

            pred = rnn(sen_batch, l_batch)
            loss = criteon(pred, label_batch)
            accuracy = acc(pred, label_batch)
            
            avg_loss.append(loss.data.item())
            avg_acc.append(accuracy.item())
            pbar.set_postfix(Loss = loss.data.item(), Accuracy = accuracy.item())

    pbar.close()
    return (avg_loss, avg_acc)


def train_model(vector_path, train_path, test_path, max_len, len_name, batch_size, lr, epoch_num, num_layers = 1, features = 2, bidir = False, save_path = None):
    '''
    训练模型
    vector_path: 预训练词向量路径
    train_path: 训练数据路径
    test_path：测试数据路径
    max_len: 最长句子长度
    len_name: 在训练数据中长度列的列名
    batch_size: batch大小
    lr: 学习率
    epoch_num: 训练轮数
    num_layers: lstm层数
    features: 使用最后几个时间步的hidden state
    save_path: 训练完成后保存路径, .pth格式
    '''
    print('start...')
    device = set_device()
    vector = KeyedVectors.load_word2vec_format(vector_path,binary=False, encoding="utf8",  unicode_errors='ignore')
    print('vector loaded')
    train_data = dataset(train_path, vector, max_len, len_name)
    test_data = dataset(test_path, vector, max_len, len_name)
    rnn = model(300, max_len, num_layers = num_layers, features = features, bidir = bidir)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criteon = torch.nn.BCEWithLogitsLoss().to(device)
    rnn.to(device)

    for epoch in range(epoch_num):
        print(f'epoch {epoch}:\n')
        print('train:\n')
        (avg_loss, avg_acc) = train(rnn, train_loader, optimizer, criteon, device)
        print(f'avg_loss = {np.array(avg_loss).mean()}, avg_acc = {np.array(avg_acc).mean()}')
        print('test:\n')
        (avg_loss, avg_acc) = evaluate(rnn, test_loader, criteon, device)
        print(f'avg_loss = {np.array(avg_loss).mean()}, avg_acc = {np.array(avg_acc).mean()}')
    
    if save_path:
        torch.save(rnn, save_path)

    return rnn


def main_train():
    '''
    训练流程
    '''
    config = {}
    config['vector_path'] = './data/sgns.sogou.bigram' 
    config['len_name'] = 'len_sg'
    config['train_path'] = './data/train_0.csv'
    config['test_path'] = './data/test_0.csv'
    config['max_len'] = 208
    config['features'] = 3
    config['batch_size'] = 256
    config['lr'] = 3e-4
    config['epoch_num'] = 8
    config['save_path'] = './data/set0_f3.pth'
    rnn = train_model(**config)

    config['train_path'] = './data/train_1.csv'
    config['test_path'] = './data/test_1.csv'
    config['features'] = 5
    config['save_path'] = './data/set1_f5.pth'
    rnn = train_model(**config)

    config['train_path'] = './data/train_2.csv'
    config['test_path'] = './data/test_2.csv'
    config['features'] = 7
    config['save_path'] = './data/set2_f7.pth'
    rnn = train_model(**config)


def sent_classi(sentence, vector, model, device = None, cut = True):
    '''
    情感分析
    sentence: 待分析句子
    vector: 预训练词向量
    model: 已训练模型存储路径
    cut: 是否需要调用jieba分词
    ''' 
    if device is None:
        device = set_device()
    vec = convert_input.cut_sentence(sentence, vector, cut)
    l = torch.tensor(vec.shape[0]).unsqueeze(0).to(device)
    vec = torch.tensor(vec).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    pred = model(vec, l)
    return pred[0]


def get_sentiment(sentence, vector_path = './data/sgns.sogou.bigram', model_paths = ['./data/set0_f3.pth', './data/set1_f5.pth', './data/set2_f7.pth']):
    '''
    综合多个模型进行情感分析,将sigmoid之后分数取平均再取整
    vector_path: 预训练词向量路径或预训练词向量
    sentence: 待分析的句子
    model_paths: 模型存储路径列表
    '''
    device = set_device()
    if type(vector_path) == str:
        vector = KeyedVectors.load_word2vec_format(vector_path,binary=False, encoding="utf8",  unicode_errors='ignore')
    else:
        vector = vector_path
    score_sum = 0
    for path in model_paths:
        rnn = torch.load(path)
        score = torch.sigmoid(sent_classi(sentence, vector, rnn, device)).data.item()
        score_sum = score_sum + score
    score_sum = score_sum/len(model_paths)
    pred = np.round(score_sum)
    return pred
