import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties


def visual_entity_number():
    df_province = pd.read_csv('KG/entity/province.csv', header=None)
    province_num = df_province.shape[0]

    df_station = pd.read_csv('KG/entity/station.csv', header=None)
    station_num = df_station.shape[0]

    df_train_no = pd.read_csv('KG/entity/train_no.csv', header=None)
    train_no_num = df_train_no.shape[0]

    df_train_node = pd.read_csv('KG/entity/train_node_站点信息.csv', header=None)
    train_node_num = df_train_node.shape[0]

    df_train_info = pd.read_csv('KG/entity/train_node_详细信息.csv', header=None)
    train_info_num = df_train_info.shape[0]

    df_train_type = pd.read_csv('KG/entity/train_type.csv', header=None)
    train_type_num = df_train_type.shape[0]

    # 中文乱码的处理
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画布大小
    # plt.figure(figsize=(13, 10))

    # 字体大小
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    x = np.array(["Province", "Station", "TrainNo", "TrainNode", "TrainInfo", "TrainType"])
    y = np.array([province_num, station_num, train_no_num, train_node_num, train_info_num, train_type_num])
    for xt, yt in enumerate(y):
        plt.text(yt + 0.2, xt, '%s' % yt, va='center')

    plt.title("实体数量统计直方图", fontsize=16)

    # plt.ylabel("Entity Type")
    # plt.xlabel("Quantity")
    plt.barh(x, y)

    plt.xticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000])
    plt.show()


def visual_relationship_number():
    df1 = pd.read_csv('KG/relationship/station_and_province.csv', header=None)
    num1 = df1.shape[0]

    df2 = pd.read_csv('KG/relationship/train_no_and_train_type.csv', header=None)
    num2 = df2.shape[0] + 1

    df3 = pd.read_csv('KG/relationship/train_no_and_train_node.csv', header=None)
    num3 = df3.shape[0]

    df4 = pd.read_csv('KG/entity/train_node_途径.csv', header=None)
    num4 = df4.shape[0]

    df_info = pd.read_csv('KG/entity/train_node_详细信息.csv', header=None)
    num_info = df_info.shape[0]

    num5 = int(num_info / 6)
    print(num5)

    # 中文乱码的处理
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # Times New Roman
    plt.rcParams['axes.unicode_minus'] = False

    # 画布大小
    # plt.figure(figsize=(13, 10))

    # 字体大小
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    x = np.array(["所属省份", "实例关系", "站点信息", "途径", "站点顺序", "站点性质", "到达时间", "发车时间", "行驶时长", "行驶距离"])
    y = np.array([num1, num2, num3, num4, num5, num5, num5, num5, num5, num5])
    for xt, yt in enumerate(y):
        plt.text(yt + 0.2, xt, '%s' % yt, va='baseline')

    plt.title("关系数量统计直方图", fontsize=16)
    # plt.ylabel("Entity Type")
    # plt.xlabel("Quantity")

    plt.barh(x, y)
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000])
    plt.show()


def visual_all_question_template():
    file_path = r"E:\project\TrainKGQA\QA\question\all_question_template.xlsx"
    data = pd.read_excel(file_path, sheet_name=0)
    # ss = data.head()
    # print(ss)

    question_template_no = []  # 每种问题数量
    for column_no in data.columns:
        count_row = 0
        for row in data[column_no]:
            if type(row) is not float:
                count_row = count_row + 1

        question_template_no.append(count_row)

    print(question_template_no)

    total = np.array(question_template_no).sum()
    print(total)

    # 绘图部分
    plt.figure(figsize=(8, 4))
    plt.rcParams['font.sans-serif'] = ['Times New Roman'] # SimSun Times New Roman
    fontcn = {'family': 'SimSun', 'size': 12}  # 1pt = 4/3px  10.5=五号 12=小四
    fonten = {'family': 'Times New Roman', 'size': 10.5}

    x = data.columns.to_list()
    y = question_template_no
    for xt, yt in enumerate(y):
        plt.text(xt + 0.6, yt + 1.0, '%s' % yt, va='center', fontdict=fonten)

    # plt.title("问题种子模板数量统计直方图", fontsize=12)
    plt.title("问题种子模板数量统计直方图", fontdict=fontcn)
    plt.bar(x, y, width=0.8)
    plt.xticks(x)
    plt.show()


if __name__ == '__main__':
    # visual_entity_number()
    # visual_relationship_number()
    visual_all_question_template()
