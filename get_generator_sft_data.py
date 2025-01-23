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

generator_sft_list = []
for i in tqdm(range(len(top_k_list))):
    cur_que_ans_docs = top_k_list[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_docs = [doc['content'] for doc in cur_que_ans_docs['top_k_docs']]
    messages = get_messages(question=question, top_docs=top_docs, top_k=len(top_docs))
    messages.append({'role': 'assistant', 'content': str(golden_answers[i])})
    generator_sft_list.append({'messages': messages})

save_results_path = "/root/paddlejob/workspace/env_run/rag/data"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/train_sft_generator_data.json', 'w') as file:
    json.dump(generator_sft_list, file)

print(1)