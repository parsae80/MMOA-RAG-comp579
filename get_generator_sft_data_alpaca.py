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

def re_replace(old_string):
    pattern = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5（）().,，!！？；;:：、\n -]')
    new_string = pattern.sub('', old_string)

    return new_string

def get_prefix_role_prompt():

    return [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        # {'role': 'user', 'content': "Before giving you question and documents, I show you somes examples.\n\nQuestion: Who was the first person killed in a car accident\nAnswer: Bridget Driscoll\n\nQuestion: Are both The New Pornographers and Kings of Leon American rock bands?\nAnswer: no\n\nQuestion: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?\nAnswer: 6.213 km long\n\nQuestion: Which was the first European country to abolish capital punishment?\nAnswer: Norway\n\nPlease answer the question briefly like above examples."},
        # {'role': 'assistant', 'content': 'Okay, please provide the question and the corresponding documents.'}
    ]

def get_post_role_prompt():
    return {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer and nothing else."}

def get_messages(question, top_docs, top_k=5):
    messages = get_prefix_role_prompt()
    messages.append(
        {'role': 'user', 'content': 'Question is: {}'.format(str(question))})
    messages.append(
        {'role': 'assistant', 'content': 'Received Question.'})
    for j in range(top_k):
        doc_id = j
        content = top_docs[j]
        messages.append(
            {'role': 'user', 'content': 'Document {}: {}'.format(str(doc_id), str(content))})
        messages.append(
            {'role': 'assistant', 'content': 'Received Document {}.'.format(str(j))})
    messages.append(get_post_role_prompt())
    return messages

top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/train_top_k_docs.jsonl'

model_name = 'Meta-Llama-3-8B-Instruct'
generator_model_path = '/home/chenyiqun/rag/meta-llama/{}'.format(model_name)

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

# get sft data for generator
questions = [top_k_list[k]['question'] for k in range(len(top_k_list))]
golden_answers = [top_k_list[k]['answer'] for k in range(len(top_k_list))]
predict_answers = []

import random
random.shuffle(top_k_list)
top_k_list1 = top_k_list[:len(top_k_list)-3610]
top_k_list2 = top_k_list[len(top_k_list)-3610:]
print('length:', len(top_k_list1), len(top_k_list2))


# top_k_list1
generator_sft_list = []
for i in tqdm(range(len(top_k_list1))):
    cur_que_ans_docs = top_k_list1[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_k_docs = cur_que_ans_docs['top_k_docs']

    temp_dict = {}
    temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."

    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))
    input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
    temp_dict['input'] = input_content

    temp_dict['output'] = '**' + str(answer) + '**'
    temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."

    history = []
    temp_dict['history'] = history

    if i == 2:
        print(temp_dict)

    generator_sft_list.append(temp_dict)

save_results_path = "/root/paddlejob/workspace/env_run/rag/data"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/train_sft_generator_data_debug1.json', 'w') as file:
    json.dump(generator_sft_list, file)


# top_k_list2
generator_sft_list = []
for i in tqdm(range(len(top_k_list2))):
    cur_que_ans_docs = top_k_list2[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_k_docs = cur_que_ans_docs['top_k_docs']

    temp_dict = {}
    temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."

    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))
    input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
    temp_dict['input'] = input_content

    temp_dict['output'] = '**' + str(answer) + '**'
    temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."

    history = []
    temp_dict['history'] = history

    if i == 2:
        print(temp_dict)

    generator_sft_list.append(temp_dict)

save_results_path = "/root/paddlejob/workspace/env_run/rag/data"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/train_sft_generator_data_debug2.json', 'w') as file:
    json.dump(generator_sft_list, file)

print(1)