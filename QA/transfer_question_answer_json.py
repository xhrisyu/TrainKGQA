import json
import pandas as pd

result = {"train": [], "dev": [], "test": []}

train_list = []
dev_list = []
test_list = []

for question_type in [int(i) for i in
                      open("config/used_question_type.txt", "r", encoding="utf-8").readline().split(" ")]:
    df = pd.read_csv('data/qa{}.tsv'.format(question_type), sep='\t', encoding='utf-8', header=None)

    # 划分数据集(训练集:验证集:测试集 = 8:2:2)
    df_train = df.sample(frac=0.66, replace=False, random_state=0, axis=0)
    df_train_rest = df[~df.index.isin(df_train.index)]

    df_dev = df_train_rest.sample(frac=0.50, replace=False, random_state=0, axis=0)
    df_test = df_train_rest[~df_train_rest.index.isin(df_dev.index)]

    for index, row in df_train.iterrows():
        one_group = {
            "question_type": row[1],
            "question": row[2],
            "question_template": row[3],
            "topic_entity": row[4][2:-2].split("', '"),
            "cypher": row[5],
            "answer_type": row[6],
            "answer": row[7][2:-2].split("', '")
        }
        train_list.append(one_group)

    for index, row in df_dev.iterrows():
        one_group = {
            "question_type": row[1],
            "question": row[2],
            "question_template": row[3],
            "topic_entity": row[4][2:-2].split("', '"),
            "cypher": row[5],
            "answer_type": row[6],
            "answer": row[7][2:-2].split("', '")
        }
        dev_list.append(one_group)

    for index, row in df_test.iterrows():
        one_group = {
            "question_type": row[1],
            "question": row[2],
            "question_template": row[3],
            "topic_entity": row[4][2:-2].split("', '"),
            "cypher": row[5],
            "answer_type": row[6],
            "answer": row[7][2:-2].split("', '")
        }
        test_list.append(one_group)

id = 1
for i in train_list:
    i["id"] = id
    id += 1
for i in dev_list:
    i["id"] = id
    id += 1
for i in test_list:
    i["id"] = id
    id += 1

result["train"] = train_list
result["dev"] = dev_list
result["test"] = test_list

with open("json/data.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

# print(json)
