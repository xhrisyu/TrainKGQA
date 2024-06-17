import pandas as pd
import sys


# 载入实体字典
def load_alias_entity(entity_label):
    file_name = 'E:/project/TrainKGQA/QA/alias_dict/' + entity_label + '.csv'

    alias_dict = []
    df = pd.read_csv(file_name, header=None)
    for index, row in df.iterrows():
        row = row.tolist()
        # 由于每个entity的别称数量不相同，所以csv中会有空值nan，需要剔除
        row_no_nan = []
        for item in row:
            if type(item) is not float:
                row_no_nan.append(item)

        alias_dict.append(row_no_nan)

    # print(alias_dict)
    return alias_dict


def main(entity_label, entity_name=None):
    alias_dict = load_alias_entity(entity_label)
    for item in alias_dict:
        if entity_name in item:
            print(item[0])
            return item[0]
    return None


if __name__ == '__main__':
    # sys.argv[1]对应的是java代码中的args1数组中的第3个值，sys.argv[2]对应第4个
    main(sys.argv[1], sys.argv[2])
