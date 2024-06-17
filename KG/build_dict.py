import numpy as np
import pandas as pd


# 建立字典，车站station，省份province，车次类型train_type
def build_dict():
    dictionary = {}
    index = 10001

    # 车站Station
    with open('entity/station.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dictionary[line[:-1]] = index
            print([line[:-1], index])
            index = index + 1

    # 省份Province
    with open('entity/province.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dictionary[line[:-1]] = index
            print([line[:-1], index])
            index = index + 1

    # 车次类型TrainType
    with open('entity/train_type.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dictionary[line[:-1]] = index
            print([line[:-1], index])
            index = index + 1

    # # 站点详细信息
    # df = pd.read_csv('relationship/train_node_详细信息.csv', encoding='utf-8', header=None)
    # for no, row in df.iterrows():
    #     dictionary[row[2]] = index
    #     print([line[:-1], index])
    #     index = index + 1

    # 车次号TrainNo
    with open('entity/train_no.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dictionary[line[:-1]] = str(line[:-1])
            print([line[:-1], line[:-1]])

    # 站点ID TrainNode
    df = pd.read_csv('entity/train_node_站点信息.csv', encoding='utf-8', header=None)
    for no, row in df.iterrows():
        dictionary[row[2]] = str(row[2])
        print([row[2], row[2]])

    # 保存字典
    np.save('dict/dictionary.npy', dictionary)


build_dict()
