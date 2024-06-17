from itertools import islice
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
import numpy as np
import csv


# 收集所有详细站点信息网页链接，保存省份实体Province，车站实体Station
def get_provinces_stations_links():
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

    # 提取省份链接、省份名
    province_link_list = []
    province_name_set = set()
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
        province_name_set.add(name)

    # 遍历每个省份链接，提取车站链接、车站名
    station_link_list = []
    station_name_set = set()
    for index, link in enumerate(province_link_list):
        province_html = requests.get(link, headers=headers)
        station_data = province_html.text.encode(html.encoding).decode('utf-8')
        s = BeautifulSoup(station_data, 'lxml')
        all_station = s.select('.bd > .table > table > tr > td > a')  # 定位
        # 遍历该省份中的每个车站
        for station in all_station:
            station_link_list.append(url + station["href"][7:])
            station_name_set.add(str(station.next_element).replace(" ", "") + '站')  # 删除空格
    station_name_set.add("椑木镇站")
    station_name_set.add("步行街站")
    station_name_set.add("香港红磡站")

    # 将爬取链接写入txt文件
    with open('entity/station_link.txt', 'w', encoding='utf-8') as f:
        for item in station_link_list:
            f.write(item + '\r')
            print(item)

    # 一维list变二维list，便于存入csv文件
    province_name_set = np.array(list(province_name_set)).reshape(len(province_name_set), 1)
    station_name_set = np.array(list(station_name_set)).reshape(len(station_name_set), 1)

    # 写入省份名
    province_name_set = sorted(province_name_set)
    with open('entity/province.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in province_name_set:
            writer.writerow(row)

    # 写入车站名
    station_name_set = sorted(station_name_set)
    with open('entity/station.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in station_name_set:
            writer.writerow(row)


# 爬取所有列车号TrainNo
def get_train_no():
    # 从文件中读取链接
    url_list = []
    with open('entity/station_link.txt', 'r') as f:
        for line in f.readlines():
            line = line[:-1]  # 不读换行符
            url_list.append(line)

    # Selenium模拟浏览器行为，可以动态爬取表格信息
    browser = webdriver.Chrome()  # 打开chrome浏览器
    browser.maximize_window()  # 最大化窗口
    wait = WebDriverWait(browser, 2)  # 等待加载10s

    # 文件存储车次号（有重复）
    train_no_raw = []

    for url in url_list:
        # 跳转网页，并停顿等待加载
        browser.get(url)
        time.sleep(0.5)

        # 找到表格信息，并等待表格加载完毕
        table_xpath = '/html/body/div/div[2]/div[2]/div[2]/div[2]/div[1]/table'
        wait.until(EC.presence_of_element_located((By.XPATH, table_xpath)))

        # 通过css选择器，找到表信息
        soup = BeautifulSoup(browser.page_source, 'lxml')
        table_css_select = 'body > div > div.container > div.content > div.mod-panel > div.bd > div:nth-child(1) > table'
        table_content = soup.select(table_css_select)[0]

        # 表信息中找到车次号，并添加到set集合中，去重
        all_train_no = table_content.select('table > tbody > tr > td > a > b')  # 定位到td标签
        for item in all_train_no:
            train_no_raw.append(item.next_element)
            # file.write(item.next_element + '\n')
        print('已完成:' + url)

    # 将车次号去重，排序，写入csv文件
    train_no_set = set()
    for item in train_no_raw:
        train_no_set.add(item)

    train_no_set = sorted(train_no_set)

    train_no_list = np.array(train_no_set).reshape(len(train_no_set), 1)
    with open('entity/train_no.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in train_no_list:
            writer.writerow(row)


# 爬取列车所有信息TrainInfo
def get_train_info():
    # 所有车次url
    url = 'https://qq.ip138.com/train/'
    train_no_list = []
    with open('entity/train_no.csv', 'r') as f:
        for line in f.readlines():
            line = line[:-1]  # 不读换行符
            train_no_list.append(line)

    # Selenium模拟浏览器行为，可以动态爬取表格信息
    browser = webdriver.Chrome()  # 打开chrome浏览器
    browser.maximize_window()  # 最大化窗口
    wait = WebDriverWait(browser, 2)  # 等待加载10s

    for item in train_no_list:
        # 跳转网页，并停顿等待加载
        link = url + item + '.htm'
        browser.get(link)
        time.sleep(0.3)

        # 找到表格信息，并等待表格加载完毕
        table_xpath = '//*[@id="stationInfo"]'
        wait.until(EC.presence_of_element_located((By.XPATH, table_xpath)))

        # 通过css选择器，找到表信息
        soup = BeautifulSoup(browser.page_source, 'lxml')
        table_css_select = '#stationInfo'
        table_content = soup.select(table_css_select)[0]

        # print(table_content)

        df = pd.DataFrame()
        # 读取表格数据
        df_table = pd.read_html(str(table_content))[0]
        # 写入df数据框中
        df = df.append(df_table, ignore_index=True)
        # DataFrame保存为csv
        df.to_csv('../../data/raw/{}.csv'.format(item), index=False)  # 不保留索引

        print('已完成车次' + item + '\t' + link)


# 建立车次信息关系(所有关系分开存储)
def build_train_info():
    # 读取车次号
    train_no_list = []
    with open('entity/train_no.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_no_list.append(line[:-1])  # 不读换行符

    relation_file_1 = open('entity/train_node_站点信息.csv', 'w', encoding='utf-8', newline='')
    writer1 = csv.writer(relation_file_1)
    relation_file_2 = open('entity/train_node_途径.csv', 'w', encoding='utf-8', newline='')
    writer2 = csv.writer(relation_file_2)
    relation_file_3 = open('entity/train_node_详细信息.csv', 'w', encoding='utf-8', newline='')
    writer3 = csv.writer(relation_file_3)

    for train_no in train_no_list:
        # 依次读取 '车次.csv' 文件
        csv_file = csv.reader(open('data/raw/{}.csv'.format(train_no), 'r', encoding='utf-8'))
        # 计算文件有几行，即该车次经过站数量
        length = len(open('data/raw/{}.csv'.format(train_no), 'r', encoding='utf-8').readlines()) - 1
        # 遍历每行数据
        for i, item in enumerate(islice(csv_file, 1, None)):  # 不读第一行列名
            station_seq = item[0]  # 站点顺序
            station_name = str(item[1]).replace(" ", "")  # 站名
            time_arrive = item[2] if i > 0 else item[3]
            time_depart = item[3]  # 出发时间
            time_travel = item[4]  # 行驶时长
            miles = item[5] if i > 0 else 0  # 行驶距离（公里）
            node_id = '{}-{}'.format(train_no, station_seq)  # 结点ID

            # 逗号分割，按行写入csv文件
            writer1.writerow([train_no, '站点信息', node_id])
            writer2.writerow([node_id, '途径', station_name + '站'])
            writer3.writerow([node_id, '站点顺序', station_seq])
            if i + 1 != length:
                writer3.writerow([node_id, '站点性质', 0])  # 1表示终点站
            else:
                writer3.writerow([node_id, '站点性质', 1])
            writer3.writerow([node_id, '到达时间', time_arrive])
            writer3.writerow([node_id, '发车时间', time_depart])
            writer3.writerow([node_id, '行驶时长', time_travel])
            writer3.writerow([node_id, '行驶距离', miles])

    relation_file_1.close()
    relation_file_2.close()
    relation_file_3.close()


# 建立车次信息关系
def build_train_info_without_null():
    # 读取车次号
    train_no_list = []
    with open('entity/train_no.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_no_list.append(line[:-1])  # 不读换行符

    relation_file_1 = open('entity/train_node_站点信息.csv', 'w', encoding='utf-8', newline='')
    writer1 = csv.writer(relation_file_1)
    relation_file_2 = open('entity/train_node_途径.csv', 'w', encoding='utf-8', newline='')
    writer2 = csv.writer(relation_file_2)
    relation_file_3 = open('entity/train_node_详细信息.csv', 'w', encoding='utf-8', newline='')
    writer3 = csv.writer(relation_file_3)

    for train_no in train_no_list:
        # 依次读取 '车次.csv' 文件
        csv_file = csv.reader(open('data/raw2/{}.csv'.format(train_no), 'r', encoding='utf-8'))
        # 计算文件有几行，即该车次经过站数量
        length = len(open('data/raw2/{}.csv'.format(train_no), 'r', encoding='utf-8').readlines()) - 1
        # 遍历每行数据
        for i, item in enumerate(islice(csv_file, 1, None)):  # 不读第一行列名
            station_seq = item[0]  # 站点顺序
            station_name = str(item[1]).replace(" ", "")  # 站名
            time_arrive = item[2]  # 到达时间
            time_depart = item[3]  # 出发时间
            time_travel = item[4]  # 行驶时长
            miles = item[5]  # 行驶距离（公里）
            node_id = '{}-{}'.format(train_no, station_seq)  # 结点ID

            # 逗号分割，按行写入csv文件
            writer1.writerow([train_no, '站点信息', node_id])
            writer2.writerow([node_id, '途径', station_name + '站'])
            writer3.writerow([node_id, '站点顺序', station_seq])
            if i + 1 != length:
                writer3.writerow([node_id, '站点性质', 0])  # 1表示终点站
            else:
                writer3.writerow([node_id, '站点性质', 1])
            writer3.writerow([node_id, '到达时间', time_arrive])
            writer3.writerow([node_id, '发车时间', time_depart])
            writer3.writerow([node_id, '行驶时长', time_travel])
            writer3.writerow([node_id, '行驶距离', miles])

    relation_file_1.close()
    relation_file_2.close()
    relation_file_3.close()


# get_provinces_stations_links()
# get_train_no()
# get_train_info()
build_train_info()
