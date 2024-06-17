import csv
import random
from py2neo import Graph
import pandas as pd
import numpy as np
import re


# 载入cypher模板
def load_cypher_template(question_type):
    data = pd.read_csv("cypher/cypher_template.tsv", sep='\t', header=None)
    for index, row in data.iterrows():
        no = row[0]
        cypher = row[1]
        answer_type = row[2]
        if str(no) == str(question_type):
            return cypher, answer_type
    print("没有搜索到该问题的模板")
    return None


# 载入问题模板
def load_question_template(question_type):
    question_list = []
    df = pd.read_csv("question/question_template.csv")
    for index, row in df.iterrows():
        if row['type'] == question_type:
            question_list.append(row['template'])

    return question_list


# 载入实体字典
def load_entity_dict(entity_label):
    file_name = 'dict_for_dataset/{}_for_dataset.csv'.format(entity_label)
    entity_dict = []
    df = pd.read_csv(file_name, header=None)
    for index, row in df.iterrows():
        row = row.tolist()
        # 由于每个entity的别称数量不相同，所以csv中会有空值nan，需要剔除
        row_no_nan = []
        for item in row:
            if type(item) is not float:
                row_no_nan.append(item)
        entity_dict.append(row_no_nan)

    return entity_dict


# 搜索答案
def search_answer(graph, cql):
    result = graph.run(cql)
    # return result.data()
    return result.to_series().tolist()  # 返回list格式


# 车次类型转换
def transfer_train_type_regex(train_type):
    if train_type == '高铁':
        return r'G\\\\d{1,4}'
    elif train_type == '动车':
        return r'[C,D]\\\\d{1,4}'
    elif train_type in ['直达', '直达特快', '直特']:
        return r'Z\\\\d{1,4}'
    elif train_type == '普快':
        return r'\\\\d{4}'
    elif train_type == '特快':
        return r'T\\\\d{1,4}'
    elif train_type == '快速':
        return r'[K,S,Y]\\\\d{1,4}'
    else:
        return r'.*'


# 检查topic_entity是否有相同的实体
def check_have_same_item(entity_list):
    if len(entity_list) == len(set(entity_list)):
        return False
    return True


if __name__ == '__main__':

    # 链接neo4j数据库
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "yzxneo4j"))

    # 正则表达式，匹配question中的{}
    pattern = re.compile(r'{\w+}')

    # 加载实体
    province_list = load_entity_dict('province')
    province_max = len(province_list)
    station_list = load_entity_dict('station')
    station_max = len(station_list)
    train_no_list = load_entity_dict('train_no')
    train_no_max = len(train_no_list)
    train_type_list = load_entity_dict('train_type')
    train_type_max = len(train_type_list)

    # 如果有该问题的问题模板，则执行
    for question_type in [int(i) for i in
                          open("config/used_question_type.txt", "r", encoding="utf-8").readline().split(" ")]:
        # if cypher_template is not None:

        # 读取该类型的所有问题
        question_template_list = load_question_template(question_type)

        # 待写入问题-答案对 tsv文件
        qa_file = open('data/qa{}.tsv'.format(question_type), 'w', encoding='utf-8', newline='')
        writer = csv.writer(qa_file, delimiter='\t')

        # 读取该问题类型的cypher查询语句模板，以及答案类型
        cypher_template, answer_type = load_cypher_template(question_type)
        print("问题{}的cypher模板:".format(question_type) + cypher_template + '\n\n')

        # 有效问题-答案对数量
        count = 1
        while count <= 500:

            # 每个问题进行实体替换
            for question in question_template_list:
                # 问题模板
                question_template = question
                print('问题模板:' + question_template)

                # cypher模板
                cypher = cypher_template

                # 查询此条question中所有{实体}
                entities = pattern.findall(question)

                # 替换的实体
                topic_entity = []

                # 对每个{entity}进行对应的替换，取值范围['{province}','{station}','{train_type}','{train_no}']
                for entity in entities:
                    new_entity_list = []
                    # 随机抽取一个实体
                    if entity == '{province}':
                        i = random.randint(0, province_max - 1)
                        new_entity_list = province_list[i]
                    elif entity == '{station}':
                        i = random.randint(0, station_max - 1)
                        new_entity_list = station_list[i]
                    elif entity == '{train_type}':
                        i = random.randint(0, train_type_max - 1)
                        new_entity_list = train_type_list[i]
                    elif entity == '{train_no}':
                        i = random.randint(0, train_no_max - 1)
                        new_entity_list = train_no_list[i]
                    else:
                        ex = Exception('问题"' + question + '"中代替换实体不符合命名要求:' + entity)
                        raise ex

                    # 列表第一项为neo4j数据库中的实体名
                    topic_entity.append(new_entity_list[0])

                    # 从实体中，随机选择一个别名
                    j = random.randint(0, len(new_entity_list) - 1)
                    entity_other_name = new_entity_list[j]

                    # 将新实体填入问题
                    question = re.sub(entity, entity_other_name, question, count=1)

                    # 将新实体填入cypher模板
                    if entity == '{train_type}':  # 车次类型需要转换
                        cypher = re.sub(entity, transfer_train_type_regex(entity_other_name), cypher, count=1)
                    else:
                        cypher = re.sub(entity, new_entity_list[0], cypher, count=1)

                print('查询语句:' + cypher)
                print('问题:' + question)

                # 查询答案，返回结果为list
                answer = search_answer(graph, cypher)
                if answer and not check_have_same_item(topic_entity):  # 如果有答案，则输出答案，数量+1
                    print('第{}个答案:'.format(count) + str(answer))

                    writer.writerow(
                        [count, question_type, question, question_template, topic_entity, cypher, answer_type, answer])

                    count = count + 1

                print('\n')

        qa_file.close()
