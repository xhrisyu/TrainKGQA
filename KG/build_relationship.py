import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

dictionary = np.load('dict/dictionary.npy', allow_pickle=True).item()


# <Station, 所属省份, Province>
def station_and_province():
    url = 'https://qq.ip138.com/train/'
    headers = {
        'Cookie': '_uab_collina=163921824960707587587211; JSESSIONID=31AFF22FC214975D0E4A3D181A61479C; BIGipServerotn=4040622346.64545.0000; BIGipServerpassport=786956554.50215.0000; guidesStatus=off; highContrastMode=defaltMode; cursorStatus=off; RAIL_EXPIRATION=1639514488347; RAIL_DEVICEID=pAxyUJ8z3Rg9-NEvZfh_nHBMvUVJ3h2lJyAfwTNFV2wh4xNlowjSB5YUylxsaTNIbBf_nieZYfgmHazriTYdeLTDy8J_D2TTcOLoKrK6GNM0_dwpgSIwIIkz3HGwaH3oCPFxQ1DzKi0J-XJeVWzj7R8bxfGvfDZ8; route=495c805987d0f5c8c84b14f60212447d; _jc_save_fromStation=%u6210%u90FD%2CCDW; _jc_save_toStation=%u91CD%u5E86%2CCQW; _jc_save_toDate=2021-12-11; _jc_save_wfdc_flag=dc; _jc_save_fromDate=2021-12-12',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                      '(KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }
    html = requests.get(url, headers=headers)
    data = html.text.encode(html.encoding).decode('utf-8')
    soup = BeautifulSoup(data, 'lxml')
    all_province = soup.select('.table > table > tr > td > a')  # 定位

    # 提取省份链接，省份名
    province_link_list = []
    province_name_list = []
    for item in all_province:
        province_link_list.append(url + item["href"][7:])
        name = item.next_element
        if name in ['北京', '上海', '天津', '重庆']:
            name = name + '市'
        elif name in ['香港', '澳门']:
            name = name + '特别行政区'
        elif name in ['内蒙古', '西藏']:
            name = name + '自治区'
        elif name == '新疆':
            name = '新疆维吾尔自治区'
        elif name == '广西':
            name = '广西壮族自治区'
        elif name == '宁夏':
            name = '宁夏回族自治区'
        else:
            name = name + '省'
        province_name_list.append(name)

    station_dict = {}  # 车站-所属省份 字典
    # 遍历每个省份链接
    for index, link in enumerate(province_link_list):
        province_html = requests.get(link, headers=headers)
        station_data = province_html.text.encode(html.encoding).decode('utf-8')
        s = BeautifulSoup(station_data, 'lxml')
        all_station = s.select('.bd > .table > table > tr > td > a')  # 定位

        # 遍历该省份中的每个车站
        for station in all_station:
            station_name = str(station.next_element + '站').replace(" ", "")  # 消除空格
            station_dict[station_name] = province_name_list[index]

    # 将词典转为带写入csv文件的list形式 [[德阳,所属省份,四川], ...]
    snames = list(station_dict.keys())
    pnames = list(station_dict.values())
    # 将list写入csv文件
    with open('relationship/station_and_province.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for i in range(0, len(snames)):
            row = [dictionary[snames[i]], snames[i], '所属省份', dictionary[pnames[i]], pnames[i]]
            print(row)
            writer.writerow(row)

        writer.writerow([dictionary['椑木镇站'], '椑木镇站', '所属省份', dictionary['四川省'], '四川省'])
        writer.writerow([dictionary['步行街站'], '步行街站', '所属省份', dictionary['四川省'], '四川省'])
        writer.writerow([dictionary['香港红磡站'], '香港红磡站', '所属省份', dictionary['香港特别行政区'], '香港特别行政区'])


# <TrainNo, 实例关系, TrainType>
def train_no_and_train_type():
    df_train_no = pd.read_csv('entity/train_no.csv', header=None)
    result_list = []
    for index, row in df_train_no.iterrows():  # 遍历每一行
        for i in range(len(row)):  # 行中遍历每个元素
            one_train = [str(row[i]), str(row[i]), '实例关系', '', '']
            # 由车次开头字母判断车型，或使用.startswith函数
            if str(row[i])[0] == 'D':
                one_train[3] = dictionary['动车']
                one_train[4] = '动车'
            elif str(row[i])[0] == 'C':
                one_train[3] = dictionary['动车']
                one_train[4] = '动车'
            elif str(row[i])[0] == 'G':
                one_train[3] = dictionary['高铁']
                one_train[4] = '高铁'
            elif str(row[i])[0] == 'Z':
                one_train[3] = dictionary['直达']
                one_train[4] = '直达'
            elif str(row[i])[0] == 'K':
                one_train[3] = dictionary['快速']
                one_train[4] = '快速'
            elif str(row[i])[0] == 'S':
                one_train[3] = dictionary['快速']
                one_train[4] = '快速'
            elif str(row[i])[0] == 'Y':
                one_train[3] = dictionary['快速']
                one_train[4] = '快速'
            elif str(row[i])[0] == 'T':
                one_train[3] = dictionary['特快']
                one_train[4] = '特快'
            else:
                one_train[3] = dictionary['普快']
                one_train[4] = '普快'
            print(one_train)
            result_list.append(one_train)

    # 写入文件，newline参数防止出现空行
    with open('relationship/train_no_and_train_type.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in result_list:
            writer.writerow(row)


# <TrainNo, 站点信息, TrainNode>
def train_no_and_train_node():
    # 读取字典
    dict = np.load('dict/dictionary.npy', allow_pickle=True).item()

    file = open('relationship/train_no_and_train_node.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    with open('entity/train_node_站点信息.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            arr = (line[:-1]).split(',')
            row = [dict[arr[0]], arr[0], arr[1], dict[arr[2]], arr[2]]
            print(row)
            writer.writerow(row)

    file.close()


# <TrainNode, ..., TrainInfo>
def train_node_and_train_info():
    # 读取字典
    dict = np.load('dict/dictionary.npy', allow_pickle=True).item()

    file = open('relationship/train_node_and_train_info.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    with open('entity/train_node_途径.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            arr = (line[:-1]).split(',')
            row = [dict[arr[0]], arr[0], arr[1], dict[arr[2]], arr[2]]
            print(row)
            writer.writerow(row)

    with open('entity/train_node_详细信息.csv', 'r', encoding='utf-8') as f:
        index = 20001
        for line in f.readlines():
            arr = (line[:-1]).split(',')
            row = [dict[arr[0]], arr[0], arr[1], index, arr[2]]
            print(row)
            writer.writerow(row)
            index = index + 1

    file.close()


# station_and_province()
train_no_and_train_type()
# train_no_and_train_node()
# train_node_and_train_info()
