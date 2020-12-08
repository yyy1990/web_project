from flask import Flask, render_template, request
import info_extract.dependency_parsing as dp

app = Flask(__name__)


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
    return render_template('sentiment_classification.html')


@app.route('/poem_gene', methods=['GET'])
def poem_gene():
    '''
    诗歌补全
    '''
    return render_template('poem_generation.html')


@app.route('/cangtou', methods=['GET'])
def cangtou():
    '''
    藏头诗生成
    '''
    return render_template('cangtou.html')