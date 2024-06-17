import pandas as pd

# xlsx 转为 csv

question_template_list = []

for question_type in [int(i) for i in
                      open("config/used_question_type.txt", "r", encoding="utf-8").readline().split(" ")]:
    df = pd.read_excel("question/question_template.xlsx", sheet_name=str(question_type), header=None)
    for index, row in df.iterrows():
        question_template_list.append([row[0], question_type])

df_question_template = pd.DataFrame(columns=['question', 'type'], data=question_template_list)
df_question_template.to_csv('question/question_template.csv')
