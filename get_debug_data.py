import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time
import json
from normalize_answers import *
from collections import Counter
import os
import re
import random

def re_replace(old_string):
    pattern = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5（）().,，!！？；;:：、\n -]')
    new_string = pattern.sub('', old_string)

    return new_string

def get_positive_position():
    # 定义列表的长度
    list_length = 10
    # numbers of active docs
    selected_number = random.choice(range(1, list_length))
    # 初始化一个全为0的列表
    binary_list = [0] * list_length
    # 随机选择 selected_number 个不同的索引
    indices_to_set = random.sample(range(list_length), selected_number)
    # 将选定索引位置的元素置为1
    for index in indices_to_set:
        binary_list[index] = 1

    return binary_list, selected_number


top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/train_top_k_docs.jsonl'

# loading questions-answer-topk_docs data
print('*'*20)
print('loading questions-answer-topk_docs data')
start_time = time.time()
# 创建一个空列表来存放字典
top_k_list = []
# 打开文件以读取
with open(top_k_docs_path, 'r') as file:
    for line in file:
        # json.loads() 函数将每行的json字符串转化为字典
        top_k_list.append(json.loads(line))
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))


import random
random.shuffle(top_k_list)
top_k_list1 = top_k_list[:len(top_k_list)-3610]
top_k_list2 = top_k_list[len(top_k_list)-3610:]
print('length:', len(top_k_list1), len(top_k_list2))


# 保存top-k的retrieval结果
train_debug_path = '/root/paddlejob/workspace/env_run/rag/data/train_top_k_docs_debug.jsonl'
with open(train_debug_path, 'w') as file:
    for dictionary in top_k_list1:
        # json.dumps()函数将字典转化为json字符串
        # 每个字典写入一行
        file.write(json.dumps(dictionary) + '\n')
val_debug_path = '/root/paddlejob/workspace/env_run/rag/data/val_top_k_docs_debug.jsonl'
with open(val_debug_path, 'w') as file:
    for dictionary in top_k_list2:
        # json.dumps()函数将字典转化为json字符串
        # 每个字典写入一行
        file.write(json.dumps(dictionary) + '\n')


# top_k_list1 top_k_list1 top_k_list1 top_k_list1 top_k_list1
top_k_list = top_k_list1
# get sft data for generator and selector
questions = [top_k_list[k]['question'] for k in range(len(top_k_list))]
golden_answers = [top_k_list[k]['answer'] for k in range(len(top_k_list))]
predict_answers = []

# top_k_list = top_k_list[:10]

selector_and_generator_sft_list = []
for i in tqdm(range(len(top_k_list))):
    cur_que_ans_docs = top_k_list[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_k_docs = cur_que_ans_docs['top_k_docs']


    # ****************** selector ******************
    binary_list, selected_number = get_positive_position()
    active_docs_content = []
    negative_docs_content = []
    for doc_id in range(selected_number):
        doc_content = top_k_docs[doc_id]['content']
        active_docs_content.append(doc_content)
    for doc_id in range(selected_number, len(top_k_docs)):
        doc_content = top_k_docs[doc_id]['content']
        negative_docs_content.append(doc_content)
    random.shuffle(active_docs_content)
    random.shuffle(negative_docs_content)

    if i == 2:
        print('binary_list: {}, selected_number: {}'.format(binary_list, selected_number))
        print('len(active_docs_content)', len(active_docs_content))
        print('len(negative_docs_content)', len(negative_docs_content))

    selector_temp_dict = {}
    selector_temp_dict['instruction'] = "Given the Question and {} candidate Documents, output the ID of the candidate Documents (0,1,2,...,{}) that is most helpful in answering the Question. Note that it is given in the following format, for example: **4,1,0,9**".format(len(top_k_docs), len(top_k_docs))

    input_content = "Question is: {}\n".format(str(question))
    active_id, negative_id = 0, 0
    for b_id in range(len(binary_list)):
        if binary_list[b_id] == 0:
            doc_content = negative_docs_content[negative_id]
            input_content = input_content + "Document {}: {}\n".format(str(b_id), str(doc_content))
            negative_id += 1
        else:
            doc_content = active_docs_content[active_id]
            input_content = input_content + "Document {}: {}\n".format(str(b_id), str(doc_content))
            active_id += 1
    input_content = input_content + "\nNow, output the ID of the candidate Documents (0,1,2,...,{}) that is most helpful in answering the Question: {}, in the following format, for example: **4,1,0,9**".format(len(top_k_docs), str(question))
    selector_temp_dict['input'] = input_content

    selector_output = ""
    for b_id in range(len(binary_list)):
        if binary_list[b_id] == 1:
            selector_output = selector_output + '{},'.format(str(b_id))
    selector_output = selector_output[:-1]
    selector_temp_dict['output'] = '**' + selector_output + '**'

    selector_temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) that is most helpful in answering the Question in the following format, for example: **4,1,0,9**".format(len(top_k_docs), len(top_k_docs))

    history = []
    selector_temp_dict['history'] = history

    if i == 2:
        print('****************** selector ******************')
        print(selector_temp_dict)
        print('****************** selector ******************')
        print('\t')

    selector_and_generator_sft_list.append(selector_temp_dict)


    # ****************** generator ******************
    temp_dict = {}
    temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."

    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(active_docs_content)):
        # doc_content = top_k_docs[doc_id]['content']
        doc_content = active_docs_content[doc_id]
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))
    input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
    temp_dict['input'] = input_content

    temp_dict['output'] = '**' + str(answer) + '**'
    temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."

    history = []
    temp_dict['history'] = history

    if i == 2:
        print('****************** generator ******************')
        print(temp_dict)
        print('****************** generator ******************')

    selector_and_generator_sft_list.append(temp_dict)


save_results_path = "/root/paddlejob/workspace/env_run/rag/data"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/train_sft_selector_generator_data_debug.json', 'w') as file:
    json.dump(selector_and_generator_sft_list, file)
print('len(selector_and_generator_sft_list): ', len(selector_and_generator_sft_list))


print('Finished!')