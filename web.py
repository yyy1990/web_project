from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    '''
    首页目录
    '''
    return render_template('index.html')


@app.route('/info_extract', methods=['GET'])
def info_extract():
    return render_template('information_extraction.html')


@app.route('/senti_classi', methods=['GET'])
def senti_classi():
    return render_template('sentiment_classification.html')


@app.route('/poem_gene', methods=['GET'])
def poem_gene():
    return render_template('poem_generation.html')