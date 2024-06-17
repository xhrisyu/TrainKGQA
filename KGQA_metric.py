import json
import pandas as pd
import requests


def KGQA_output():
    """
    通过问答系统查询json/data.json所有问题的答案
    保存为json/output.json
    结果格式也是{'train': [{'id': , 'result':}, {'id': , 'result':}], 'dev':[], 'test':[]} 和data.json类似，只不过每个问题只有id和result
    :return:
    """

    # 载入JSON问题
    json_data = json.load(open("QA/json/data.json", "r", encoding="utf-8"))
    # 通过问答系统查询答案
    URL = "http://localhost:8080/kgqa/query"
    output = {}
    count = 1
    for group_name in ['test']:
        data = []
        for item in json_data[group_name]:
            id = item["id"]  # 问题ID
            question = item["question"]  # 问题
            r = requests.get(URL + "?question=" + question)  # GET发送请求
            r_answer = r.json()['answer']
            data.append({'id': id, 'result': r_answer})

            print(count)
            print(question)
            print(r_answer)
            print("=============================================\n\n")
            count += 1

        output[group_name] = data
        print(data)

    with open("QA/json/qa_output_1.json", "w", encoding="utf-8") as f:
        # with open("QA/json/qa_output_2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


def computeF1(origin_list, result_list):
    """
    计算 召回率、准确率、F1
    :param origin_list: 原答案
    :param result_list: 预测的答案
    :return: (recall, precision, f1)
    """
    """Assume all questions have at least one answer"""
    if len(origin_list) == 0:
        if len(result_list) == 0:
            return 1, 1, 1
        else:
            return 0, 0, 0
    """If we return an empty list recall is zero and precision is one"""
    if len(result_list) == 0:
        return 0, 1, 0

    """It is guaranteed now that both lists are not empty"""

    precision = 0
    for entity in result_list:
        if entity in origin_list:
            precision += 1
    precision = float(precision) / len(result_list)

    recall = 0
    for entity in origin_list:
        if entity in result_list:
            recall += 1
    recall = float(recall) / len(origin_list)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)

    return recall, precision, f1


def cal_qa_metric():
    """
    计算QA系统的准确率、召回率、F1-score
    :return: None
    """
    """1.载入原始问答测试数据集"""
    origin_answer_data = {}
    with open("QA/json/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data["test"]:
            q_id = item['id']
            # 这个地方的答案需要是list类型
            origin_answer_data[q_id] = {"answer": item['answer'], "question_type": item["question_type"]}

    """2.初始化字典，各个类别的准确率、召回率、F1为0"""
    question_type_list = [int(i) for i in
                          open("QA/config/used_question_type.txt", "r", encoding="utf-8").readline().split(" ")]
    metric_map = {}  # {1: {"count": , "recall": , "precision": , "f1": }, 2: {}, ...}
    for i in question_type_list:
        metric_map[i] = {}
        metric_map[i]["count"] = 0
        metric_map[i]["precision"] = 0
        metric_map[i]["recall"] = 0
        metric_map[i]["f1"] = 0

    all_precision_micro = 0
    all_recall_micro = 0
    all_f1_micro = 0
    all_count = 0
    n_correct = 0

    """3.遍历QA输出结果的每一项，计算每个问题的(precision, recall, f1)"""
    # QA输出结果格式也是{'train': [{'id': , 'result':}, {'id': , 'result':}], 'dev':} 和data.json类似，只不过每个问题只有id和result
    with open("QA/json/qa_output_2.json", "r", encoding="utf-8") as f:
        output = json.load(f)
        # 遍历QA输出结果的每一项
        for item in output['test']:
            question_type = origin_answer_data[item['id']]["question_type"]  # 问题类型
            origin = origin_answer_data[item['id']]["answer"]  # 原始答案
            result = item['result']  # 预测答案
            # print("type:{}, origin:{}, result:{}".format(question_type, origin, result))

            recall, precision, f1 = computeF1(origin, result)

            # 当前问题类型的(precision, recall, f1)
            metric_map[question_type]["count"] += 1
            metric_map[question_type]["precision"] += precision
            metric_map[question_type]["recall"] += recall
            metric_map[question_type]["f1"] += f1

            # 所有问题总体的(precision, recall, f1)
            if f1 == 1:
                n_correct += 1
            all_precision_micro += precision
            all_recall_micro += recall
            all_f1_micro += f1
            all_count += 1

    """4.每一类问题的(precision, recall, f1)求和，取算术平均，保存在CSV文件中"""
    count_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for type in metric_map:  # {1: {"count": , "recall": , "precision": , "f1": }, 2: {}, ...}
        count = metric_map[type]["count"]  # 数量
        count_list.append(count)
        precision_list.append(round(float(metric_map[type]["precision"]) / count, 4))
        recall_list.append(round(float(metric_map[type]["recall"]) / count, 4))
        f1_list.append(round(float(metric_map[type]["f1"]) / count, 4))

    df = pd.DataFrame(data=[count_list, precision_list, recall_list, f1_list], columns=question_type_list,
                      index=["count", "precision", "recall", "f1"])

    """5.总体的(precision, recall, f1)"""
    all_precision_micro = round(float(all_precision_micro) / all_count, 4)
    all_recall_micro = round(float(all_recall_micro) / all_count, 4)
    all_f1_micro = round(float(all_f1_micro) / all_count, 4)

    # print(all_precision_micro)
    # print(all_recall_micro)
    # print(all_f1_micro)

    all_precision_macro = round(sum(precision_list) / len(precision_list), 4)
    all_recall_macro = round(sum(recall_list) / len(recall_list), 4)
    all_f1_macro = round(sum(f1_list) / len(f1_list), 4)

    # print(all_precision_macro)
    # print(all_recall_macro)
    # print(all_f1_macro)

    # df["micro"] = [sum(count_list), all_precision_micro, all_recall_micro, all_f1_micro]
    df["macro"] = [sum(count_list), all_precision_macro, all_recall_macro, all_f1_macro]

    df.to_csv("QA/metric/qa_performance_2.csv")


    # print("Number of questions: " + str(all_count))
    # print("Average recall over questions: " + str(all_recall))
    # print("Average precision over questions: " + str(all_precision))
    # print("Average f1 over questions: " + str(all_f1))

    # accuracy = float(n_correct) / count
    # print("Accuracy over questions: " + str(accuracy))
    # averageNewF1 = 2 * average_recall * average_precision / (average_precision + average_recall)
    # print("F1 of average recall and average precision: " + str(averageNewF1))


# KGQA_output()
cal_qa_metric()
