from flask import Flask, jsonify, request, abort
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn
import numpy as np

HOST = 'localhost'
PORT = 5000

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("E:/project/TrainKGQA/chinese-bert-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("E:/project/TrainKGQA/chinese-bert-wwm-ext")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
question_type_list = [int(i) for i in
                      open("E:/project/TrainKGQA/QA/config/used_question_type.txt", "r",
                           encoding="utf-8").readline().split(
                          " ")]


def get_bert_emb(text, tokenizer, device, model):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    text_emb = outputs[0][0]
    text_emb = text_emb[0]
    return text_emb


# 读取的问题平均BERT词向量
def load_group_bert_emb(question_type):
    group_bert_emb = np.load('E:/project/TrainKGQA/QA/emb/bert_emb_{}.npy'.format(question_type), allow_pickle=True)
    return group_bert_emb[0]


def handleRequest(user_question):
    user_question_emb = get_bert_emb(str(user_question), tokenizer, device, model)
    question_simi = {}
    for question_type in question_type_list:
        group_question_bert_emb = load_group_bert_emb(question_type)
        simi = cos(user_question_emb, group_question_bert_emb)  # 计算相似度
        question_simi[question_type] = simi.item()

    question_simi = sorted(question_simi.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    result_type = question_simi[0][0]

    return result_type


# @app.route('/task/<question>', methods=['GET'])
@app.route('/task', methods=['GET'])
def get_task():
    question = request.args.get('question')
    result_type = handleRequest(question)
    return jsonify({'type': result_type})


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host=HOST, port=PORT)
