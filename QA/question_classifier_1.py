import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn
import json
import sys
from sklearn.metrics import precision_recall_fscore_support, f1_score, recall_score, precision_score

# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
# pytorch_model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("E:/project/TrainKGQA/chinese-bert-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("E:/project/TrainKGQA/chinese-bert-wwm-ext")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 载入问题类型
def load_question_type():
    return [int(i) for i in
            open("E:/project/TrainKGQA/QA/config/used_question_type.txt", "r", encoding="utf-8").readline().split(
                " ")]


# 载入XXX集数据，返回字典train_question={1: [question1, question2, ...], 2:[], }
def load_json_question(data_type="train"):
    # 载入问题类型
    question_type_list = load_question_type()

    # 载入JSON数据
    json_data = json.load(open("json/data.json", "r", encoding="utf-8"))

    question_dict = {}
    # 对每个question_type键创建空列表
    for question_type in question_type_list:
        question_dict[question_type] = []

    for item in json_data[data_type]:
        if item['question_type'] in question_type_list:
            question_dict[item['question_type']].append(item['question'])

    return question_dict


# 计算text的BERT词向量
def get_bert_emb(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    text_emb = outputs[0][0]
    text_emb = text_emb[0]
    return text_emb


# 保存训练集问题的BERT词向量
def save_question_group_bert_emb(question_type):
    # 读入训练数据
    train_question = load_json_question("train")

    question_emb_list = []
    # 计算当前question_type所有词向量，并取平均值
    for question in train_question[question_type]:
        question_emb = get_bert_emb(question)
        question_emb_list.append(question_emb)  # 添加到list

    count = len(train_question[question_type])  # 问题数量

    # 计算词向量平均值
    avg_emb = np.array(question_emb_list).sum() / count
    print("模板种类:{},平均emb值:{}".format(question_type, avg_emb))

    # 词向量保存至本地
    np.save("emb/bert_emb_{}".format(question_type), np.array([avg_emb]))


# 读取BERT词向量
def load_group_bert_emb(question_type):
    group_bert_emb = np.load('emb/bert_emb_{}.npy'.format(question_type), allow_pickle=True)
    return group_bert_emb[0]


# 计算用户问题与10类问题的相似度，排序，返回最优值
def cal_question_similarity(user_question):
    question_type_list = load_question_type()

    # 计算question词向量
    user_question_emb = get_bert_emb(user_question)

    question_simi = {}

    for question_type in question_type_list:
        # 加载问题组emb
        group_question_bert_emb = load_group_bert_emb(question_type)
        print("==============完成问题{}词向量载入==============".format(question_type))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        simi = cos(user_question_emb, group_question_bert_emb)  # 计算问题模板组emb 和 用户问题emb
        print("模板:{}，平均emb:{}，相似度：{}".format(question_type, group_question_bert_emb, simi.item()))
        question_simi[question_type] = simi.item()

        print("==============完成与问题{}相似度计算==============".format(question_type))

    # 键值对，按相似度降序排列
    question_simi = sorted(question_simi.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print("==============完成相似度降序排列==============")
    print("问题:{}".format(user_question))

    result_type = question_simi[0][0]
    result_simi = question_simi[0][1]
    print("分类结果:{},相似度:{}".format(result_type, result_simi))
    return result_type


def user_question_classifier(user_question):
    question_type_list = load_question_type()
    user_question_emb = get_bert_emb(user_question)
    question_simi = {}
    for question_type in question_type_list:
        group_question_bert_emb = load_group_bert_emb(question_type)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        simi = cos(user_question_emb, group_question_bert_emb)  # 计算问题模板组emb 和 用户问题emb
        question_simi[question_type] = simi.item()

    question_simi = sorted(question_simi.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    result_type = question_simi[0][0]
    # result_simi = question_simi[0][1]
    print(result_type)


def cal_classifier_output():
    """
    分类器的测试集分类结果，保存到本地
    :return: None
    """

    """载入测试集数据"""
    json_data = json.load(open("json/data.json", "r", encoding="utf-8"))  # 载入JSON数据
    test_question_dict = {}
    for item in json_data["test"]:
        test_question_dict[item['id']] = {"question": item["question"], "question_type": item['question_type'],
                                          "result_type": 0}

    """计算分类结果"""
    # test_result_list格式：{id:{question:, question_type:, result_type:}, id:{}, ...}
    for item in test_question_dict:
        question = test_question_dict[item]["question"]
        result_type = cal_question_similarity(question)
        test_question_dict[item]["result_type"] = result_type

    """保存dict为json格式到本地"""
    with open("json/classifier_output_1.json", 'w', encoding='utf-8') as f:
        json.dump(test_question_dict, f, ensure_ascii=False)


def cal_classifier_metric():
    """
    测试分类器的准确率、召回率、F1值
    :return: None
    """
    """构建实际分类list和预测分类list"""
    # classifier_output = json.load(open("json/classifier_output_1.json", "r", encoding="utf-8"))  # 载入JSON数据
    classifier_output = json.load(open("json/classifier_output_2.json", "r", encoding="utf-8"))  # 载入JSON数据
    y_true = []
    y_pred = []
    label = [1, 2, 3, 4, 5, 11, 12, 18, 19, 20]
    for item in classifier_output:
        y_true.append(classifier_output[item]["question_type"])
        y_pred.append(classifier_output[item]["result_type"])

    """计算每类的precision, recall, f1"""
    precision_list, recall_list, f1_list, count_list = precision_recall_fscore_support(y_true, y_pred, labels=label)

    precision_list = [round(i, 4) for i in precision_list]
    recall_list = [round(i, 4) for i in recall_list]
    f1_list = [round(i, 4) for i in f1_list]
    count_list = [int(i) for i in count_list]

    """计算整体的precision, recall, f1"""
    precision_macro = round(precision_score(y_true, y_pred, labels=label, average='macro'), 4)
    recall_macro = round(recall_score(y_true, y_pred, labels=label, average='macro'), 4)
    f1_macro = round(f1_score(y_true, y_pred, average='macro'), 4)

    precision_micro = round(precision_score(y_true, y_pred, labels=label, average='micro'), 4)
    recall_micro = round(recall_score(y_true, y_pred, labels=label, average='micro'), 4)
    f1_micro = round(f1_score(y_true, y_pred, average='micro'), 4)

    """结果输出"""
    print("====各个类别的准确率、召回率、F1-score====")
    print(precision_list)
    print(recall_list)
    print(f1_list)
    print(count_list)
    print("====总体的宏 准确率、召回率、F1-score====")
    print(precision_macro)
    print(recall_macro)
    print(f1_macro)
    print("====总体的微 准确率、召回率、F1-score====")
    print(precision_micro)
    print(recall_micro)
    print(f1_micro)

    """结果保存到本地"""
    df = pd.DataFrame(data=[count_list, precision_list, recall_list, f1_list], columns=label,
                      index=["count", "precision", "recall", "f1"])
    df["micro"] = [sum(count_list), precision_micro, recall_micro, f1_micro]
    df["macro"] = [sum(count_list), precision_macro, recall_macro, f1_macro]
    # df.to_csv("metric/classifier_performance_1.csv")
    df.to_csv("metric/classifier_performance_2.csv")


# def cal_classifier_metric():
#     # 载入问题类型
#     question_type_list = load_question_type()
#
#     # 载入测试数据
#     test_question_dict = load_json_question("test")
#
#     # test_result_list格式：[{question:, question_type:, result_type:}, {}, ...]
#     test_result_list = []
#     for question_type in question_type_list:
#         for question in test_question_dict[question_type]:
#             # 进行问题分类
#             result_type = cal_question_similarity(question)
#             group = {"question": question, "question_type": question_type, "result_type": result_type}
#             test_result_list.append(group)
#
#     """遍历每种问题，分别计算准确率Precision，召回率Recall，F1"""
#     # 第1类测试集中有t个问题, 在整个测试集中统计出被分类到第1类的问题的个数m，m中正确的分类的个数是n
#     # 准确率就是n/m，召回率就是n/t
#     average_recall = 0
#     average_precision = 0
#     average_f1 = 0
#     total_count = 0
#
#     calculation_list = []
#     for question_type in question_type_list:
#         count = 0
#         t = 0
#         m = 0
#         TP = 0
#         FN = 0
#         FP = 0
#
#         for item in test_result_list:
#             if item["question_type"] == question_type:  # 确保当前类型
#                 if item["result_type"] == item["question_type"]:
#                     TP += 1
#                 if item["result_type"] != item["question_type"]:
#                     FN += 1
#             elif item["result_type"] == question_type:
#                 FP += 1
#
#         for item in test_result_list:
#             if item["result_type"] == question_type:
#                 m += 1
#                 if item["result_type"] == item["question_type"]:
#                     TP += 1
#             if item["question_type"] == question_type:
#                 t += 1
#
#             count += 1
#             total_count += 1
#
#         precision = TP / m
#         recall = TP / t
#         f1 = 2 * precision * recall / (precision + recall)
#
#         average_precision += precision
#         average_recall += recall
#         average_f1 += f1
#
#         calculation_list.append([question_type, count, round(precision, 4), round(recall, 4), round(f1, 4)])
#
#     calculation_list.append(["total", total_count, round(average_precision / 10, 4), round(average_recall / 10, 4),
#                              round(average_f1 / 10, 4)])
#     precision_recall_table = pd.DataFrame(calculation_list, columns=['type', 'count', 'precision', 'recall', 'F1'])
#     precision_recall_table.to_csv("metric/classifier_performance_1.csv", index=False)
#     # print(precision_recall_table)

if __name__ == '__main__':
    cal_classifier_metric()
