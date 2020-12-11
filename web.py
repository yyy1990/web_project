from flask import Flask, render_template, request
import info_extract.dependency_parsing as dp
import sentiment.model as model
from gensim.models import KeyedVectors
import torch
import poet.poet_model as pm

app = Flask(__name__)
device = torch.device("cpu")
r_n = 4 #输出小数位后保留位数

#用于情感分析的预训练词向量
vector_path = './data/sgns.sogou.bigram' 
vector = KeyedVectors.load_word2vec_format(vector_path,binary=False, encoding="utf8",  unicode_errors='ignore')

#用于情感分析的模型路径
s1_path = './data/sentiment1.pkl'
s2_path = './data/sentiment2.pkl'
s3_path = './data/sentiment3.pkl'


@app.route('/', methods=['GET'])
def home():
    '''
    首页目录
    '''
    return render_template('index.html')


@app.route('/info_extract', methods=['GET'])
def info_extract():
    '''
    观点提取
    '''
    return render_template('information_extraction.html', input_sentence='待分析文本', result='结果')


@app.route('/info_extract', methods=['POST'])
def info_extract_result():
    '''
    观点提取结果
    '''
    sentence = request.form['input_sentence']
    try:
        result = dp.split_first(sentence)
        i = 1
        res_text = ''
        for key, value in result.items():
            res_text = res_text+str(i)+'.'+'人物：'+key+'\n'+'  观点：'+value+'\n'
            i = i+1
        return render_template('information_extraction.html', input_sentence=sentence, result=res_text)
    except:
        return render_template('index.html', message='输入错误')


@app.route('/senti_classi', methods=['GET'])
def senti_classi():
    '''
    情感分析
    '''
    return render_template('sentiment_classification.html', input_sentence = '待分析文本')


@app.route('/senti_classi', methods=['POST'])
def senti_classi_result():
    '''
    情感分析结果
    '''
    sentence = request.form['input_sentence']
    try:
        score1, res1 = get_sentiment_result(sentence, s1_path, device)
        score2, res2 = get_sentiment_result(sentence, s2_path, device)
        score3, res3 = get_sentiment_result(sentence, s2_path, device)
        avg_score = (score1+score2+score3)/3
        avg_res = score2res(avg_score)
        score1 = round(score1, r_n)
        score2 = round(score2, r_n)
        score3 = round(score3, r_n)
        avg_score = round(avg_score, r_n)
        return render_template('sentiment_classification.html', input_sentence = sentence, \
        score1 = score1, res1 = res1, score2 = score2, res2 = res2, score3 = score3, res3 = res3, \
        avg_score = avg_score, avg_res = avg_res)
    except:
        return render_template('sentiment_classification.html', message='输入错误')


def get_sentiment_result(sentence, path, device):
    '''
    获取情感分析结果
    sentence:待分析文本
    path:模型存储路径
    '''
    datas = torch.load(path)
    state_dict = datas['state_dict']
    vocab_dim = datas['vocab_dim']
    hidden_dim = datas['hidden_dim']
    num_layers = datas['num_layers']
    features = datas['features']
    bidir = datas['bidir']
    rnn = model.model(vocab_dim, hidden_dim, num_layers, features, bidir)
    rnn.load_state_dict(state_dict)
    pred = model.sent_classi(sentence, vector, rnn, device)
    score = torch.sigmoid(pred).data.item()
    res = score2res(score)
    return (score, res)


def score2res(score):
    '''
    情感分析分数转化为结果
    '''
    try:
        if score >= 0.5:
            return '正面情感'
        elif score < 0.5:
            return '负面情感'
    except:
        return '情感分数必须为数值类型'


@app.route('/poem_gene', methods=['GET'])
def poem_gene():
    '''
    诗歌补全
    '''
    return render_template('poem_generation.html', input_sentence='诗歌开头', result='生成的诗歌')


@app.route('/poem_gene', methods=['POST'])
def poem_gene_result():
    '''
    诗歌补全结果
    '''
    sentence = request.form['input_sentence']
    try:
        result = pm.generate(sentence)
        s_split = ''.join(result).split('。')
        result = ''
        for s in s_split:
            if s:
                result = result + s + '。' + '\n'
        return render_template('poem_generation.html', input_sentence=sentence, result=result)
    except:
        return render_template('poem_generation.html', message='输入错误')


@app.route('/cangtou', methods=['GET'])
def cangtou():
    '''
    藏头诗生成
    '''
    return render_template('cangtou.html', input_sentence='藏头诗开头字', result='生成的藏头诗' )


@app.route('/cangtou', methods=['POST'])
def cangtou_result():
    '''
    藏头诗结果
    '''
    sentence = request.form['input_sentence']
    try:
        result = pm.cang_tou(sentence)
        return render_template('cangtou.html', input_sentence=sentence, result=result)
    except:
        return render_template('cangtou.html', message='输入错误')


if __name__ == '__main__':
    app.run()