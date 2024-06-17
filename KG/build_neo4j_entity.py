import csv
import pandas as pd


# 建立字典，车站station，省份province，车次类型train_type
def build_neo4j_entity():
    index = 10001

    file = open('import_for_neo4j/entity.csv', 'w', encoding='utf-8', newline='')  # newline避免空行
    writer = csv.writer(file)
    writer.writerow([':ID', ':LABEL', 'name'])  # 列名 :ID,:LABEL,name,number

    # 车站
    with open('entity/station.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writer.writerow([index, 'Station', line[:-1]])
            print([index, 'Station', line[:-1]])
            index = index + 1

    # 省份
    with open('entity/province.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writer.writerow([index, 'Province', line[:-1]])
            index = index + 1
            print([index, 'Province', line[:-1]])

    # 车次类型
    with open('entity/train_type.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writer.writerow([index, 'TrainType', line[:-1]])
            print([index, 'TrainType', line[:-1]])
            index = index + 1

    # 车次号
    with open('entity/train_no.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writer.writerow([line[:-1], 'TrainNo', line[:-1]])
            print([line[:-1], 'TrainNo', line[:-1]])

    # 站点ID
    df = pd.read_csv('entity/train_node_站点信息.csv', header=None)
    for no, row in df.iterrows():
        writer.writerow([row[2], 'TrainNode', row[2]])
        print([row[2], 'TrainNode', row[2]])

    # 站点详细信息
    index = 20001
    df = pd.read_csv('entity/train_node_详细信息.csv', header=None)
    for no, row in df.iterrows():
        writer.writerow([index, 'TrainInfo', row[2]])
        print([index, 'TrainInfo', row[2]])
        index = index + 1

    file.close()


build_neo4j_entity()
