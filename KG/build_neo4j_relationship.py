import csv
import numpy as np
import pandas as pd


# 建立字典，车站station，省份province，车次类型train_type
def build_neo4j_relationship():
    file = open('import_for_neo4j/relationship.csv', 'w', encoding='utf-8', newline='')  # newline避免空行
    writer = csv.writer(file)
    writer.writerow([':START_ID', ':TYPE', ':END_ID'])  # 列名 :START_ID,:TYPE,:END_ID

    # <车站, 所属省份, 省份>
    with open('relationship/station_and_province.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1].split(',')
            writer.writerow([line[0], '所属省份', line[3]])
            print([line[0], '所属省份', line[3]])

    # <D2242, 实例关系, 动车>
    with open('relationship/train_no_and_train_type.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1].split(',')
            writer.writerow([line[0], '实例关系', line[3]])
            print([line[0], '实例关系', line[3]])

    # <D2242, 站点信息, TrainNode>
    with open('relationship/train_no_and_train_node.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1].split(',')
            writer.writerow([line[0], '站点信息', line[3]])
            print([line[0], '站点信息', line[3]])

    # <TrainNode, 途径, Station>
    # <TrainNode, 详细信息, TrainInfo>
    with open('relationship/train_node_and_train_info.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1].split(',')
            writer.writerow([line[0], line[2], line[3]])
            print([line[0], line[2], line[3]])


build_neo4j_relationship()
