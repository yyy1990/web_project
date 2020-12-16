#!/usr/bin/python
# -*- coding: gbk -*-

import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


class DictObj(object):
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp

    def __setattr__(self, name, value):
        if name == 'map':# 初始化的设置
            object.__setattr__(self, name, value)
            return
        self.map[name] = value

    def __getattr__(self, name):
        # 可直接用名字调用
        return  self.map[name]


Config = DictObj({
    'poem_path' : "./data/tang.npz",
    'tensorboard_path':'./tensorboard',
    'model_save_path':'poet_generation.pth',
    'embedding_dim':100,
    'hidden_dim':1024,
    'lr':0.001,
    'LSTM_layers':3,
    'seq_len': 48,
    'batch_size': 128,
    'epochs': 20
    })


class PoemDataSet(Dataset):
    def __init__(self,poem_path,seq_len):
        '''
        poem_path: 数据路径
        seq_len: 诗歌长度
        '''
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()
        
    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label
    
    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)
    
    def filter_space(self): 
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ): #8292代表空格
                no_space_data.append(i)
        return no_space_data
    
    def get_raw_data(self):
        datas = np.load(self.poem_path, allow_pickle = True)
        data = datas['data']
        ix2word = datas['ix2word'].item() #序号->字
        word2ix = datas['word2ix'].item() #字->序号
        return data, ix2word, word2ix


class MyPoetryModel_tanh(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel_tanh, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True,dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,vocab_size)

    def forward(self, _input, hidden=None):
        embeds = self.embeddings(_input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = _input.size()
        if hidden is None:
            h_0 = _input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = _input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output,hidden


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)): 
    '''
    topk的准确率计算
    '''
    maxk = max(topk) 
    batch_size = label.size(0)
    
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() 
    correct = pred.eq(label.view(1, -1).expand_as(pred)) 

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) 
        rtn.append(correct_k.mul_(100.0 / batch_size)) 
    return rtn



def train(epochs, train_loader, device, model, criterion, optimizer, scheduler, tensorboard_path):
    '''
    训练中使用tensorboard来绘制曲线，终端输入tensorboard --logdir=/path_to_log_dir/ --port 6006 可查看
    '''
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    for epoch in range(epochs):
        train_loss = 0.0
        train_loader = tqdm(train_loader)
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.view(-1) # 因为outputs经过平整，所以labels也要平整来对齐
            
            optimizer.zero_grad()
            outputs,hidden = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            _,pred = outputs.topk(1)
            prec1, prec2= accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)
            
            # ternsorboard 曲线绘制
            if os.path.exists(Config.tensorboard_path) == False: 
                os.mkdir(Config.tensorboard_path)    
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()
        scheduler.step()

    print('Finished Training')


def main_train(batch_size = Config.batch_size, lr = Config.lr, epochs = Config.epochs):
    poem_ds = PoemDataSet(Config.poem_path, Config.seq_len)
    word2ix = poem_ds.word2ix
    poem_loader =  DataLoader(poem_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    model = MyPoetryModel_tanh(len(word2ix),embedding_dim=Config.embedding_dim,hidden_dim=Config.hidden_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10,gamma=0.1)#学习率调整
    criterion = nn.CrossEntropyLoss()
    if os.path.exists(Config.tensorboard_path):
        shutil.rmtree(Config.tensorboard_path)  
        os.mkdir(Config.tensorboard_path)
    train(epochs, poem_loader, device, model, criterion, optimizer,scheduler, Config.tensorboard_path)
    torch.save(model, Config.model_save_path)


def generate(start_words, model_path = './data/poet.pkl', data_path = Config.poem_path, max_len = 48):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #model = torch.load(model_path)
    #载入模型
    datas = torch.load(model_path)
    state_dict = datas['state_dict']
    vocab_size = datas['vocab_size']
    embedding_dim = datas['embedding_dim']
    hidden_dim = datas['hidden_dim']
    model = MyPoetryModel_tanh(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(datas['state_dict'])

    poem_ds = PoemDataSet(data_path, Config.seq_len)
    ix2word = poem_ds.ix2word
    word2ix = poem_ds.word2ix

    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    _input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    
    #最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers*1,1,Config.hidden_dim),dtype=torch.float)
    _input = _input.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
            for i in range(max_len):#诗的长度
                output, hidden = model(_input, hidden)
                # 如果在给定的句首中，input为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    _input = _input.data.new([word2ix[w]]).view(1, 1)
                # 否则将output作为下一个input进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()#输出的预测的字
                    w = ix2word[top_index]
                    results.append(w)
                    _input = _input.data.new([top_index]).view(1, 1)
                if w == '<EOP>': # 输出了结束标志就退出
                    del results[-1]
                    break
    return results  


def cang_tou(start_words, model_path = './data/poet.pkl', data_path = Config.poem_path):
    result = []
    l = []
    for word in start_words:
        poet = generate(word, model_path, data_path)
        poet = ''.join(poet).split('。')[0]
        result.append(poet)
        l.append((len(poet)-1)/2)
    min_l = min(l)
    res_s = ''
    for res in result:
        res_s = res_s + cut_poet(res, min_l) + '\n'
    return res_s


def cut_poet(p, l):
    '''
    p:不带句号的诗词。前半句+逗号+后半句
    l:目标半句长度
    '''
    result = p.split('，')
    result = result[0][0:int(l)]+'，'+result[1][0:int(l)]+'。'
    return result


if __name__ == '__main__':
    main_train()
