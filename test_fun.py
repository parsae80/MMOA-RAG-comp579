import json

file_path = '/root/paddlejob/workspace/env_run/rag/data/query_rewrite_data/query_rewrite_hotpotqa.json'
with open(file_path, 'r', encoding='utf-8') as file:
    query_rewrite_data = json.load(file)  # 使用 json.load() 将 JSON 数据解析为字典

init_questions = []
rewritten_questions = []
query_rewrite_dict = {}

for i in range(len(query_rewrite_data)):
    init_q = query_rewrite_data[i]['source'].strip()
    rewritten_qs_list = query_rewrite_data[i]['target'].split(';')
    rewritten_qs_list = [q.strip() for q in rewritten_qs_list]
    init_questions.append(init_q)
    rewritten_questions.append(rewritten_qs_list)
    query_rewrite_dict[init_q] = rewritten_qs_list

for i in range(10):
    print(init_questions[i])
    print(rewritten_questions[i])
    print('\t')

for i in range(len(query_rewrite_data)):
    if len(rewritten_questions[i]) >= 5:
        print(len(rewritten_questions[i]))

print(query_rewrite_dict)