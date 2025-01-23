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

def get_rm_data(answers_path, top_k_docs_path):

    # loading questions-answer-topk_docs data
    print('*'*20)
    print('loading answers to get the pairwise data for reward model training.')
    start_time = time.time()
    # 创建一个空列表来存放字典
    answers_pair = []
    # 打开文件以读取
    with open(answers_path, 'r') as file:
        for line in file:
            # json.loads() 函数将每行的json字符串转化为字典
            answers_pair.append(json.loads(line))
    end_time = time.time()
    print('time consuming: {} seconds'.format(end_time - start_time))

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

    # get pair answers
    golden_answers = [answers_pair[k]['golden_answer'] for k in range(len(answers_pair))]
    predict_answers = [answers_pair[k]['predict_answer'] for k in range(len(answers_pair))]

    # top_k_list = top_k_list[:10]
    # answers_pair = answers_pair[:10]

    rm_data_list = []
    if len(top_k_list) == len(answers_pair):

        for i in tqdm(range(len(top_k_list))):
            cur_que_ans_docs = top_k_list[i]
            question = cur_que_ans_docs['question']
            answer = cur_que_ans_docs['answer']
            top_k_docs = cur_que_ans_docs['top_k_docs']

            golden_answer = golden_answers[i]
            predict_answer = predict_answers[i]

            if golden_answer == predict_answer:
                continue


            temp_dict = {}
            temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."

            input_content = "Question is: {}\n".format(str(question))
            for doc_id in range(len(top_k_docs)):
                doc_content = top_k_docs[doc_id]['content']
                input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))
            input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
            temp_dict['input'] = input_content

            temp_dict['chosen'] = '**' + str(golden_answer) + '**'
            temp_dict['rejected'] = '**' + str(predict_answer) + '**'

            if i == 2:
                print(temp_dict)

            rm_data_list.append(temp_dict)

        print('len(rm_data_list): {}, len(top_k_list): {}, rating : {}.'.format(len(rm_data_list), len(top_k_list), len(rm_data_list) / len(top_k_list)))

    else:

        print('数据不匹配, len(top_k_list)=={}, len(answers_pair)== {}'.format(len(top_k_list), len(answers_pair)))

    return rm_data_list


# 此处需要改为训练集的
answers_path_1 = '/root/paddlejob/workspace/env_run/rag/data/naive_rag/train_to_pair_2epochs_4096.jsonl'
answers_path_2 = '/root/paddlejob/workspace/env_run/rag/data/naive_rag/batch_train_to_pair_3epochs_4096.jsonl'
top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/train_top_k_docs.jsonl'

rm_list_11 = get_rm_data(answers_path_1, top_k_docs_path)
rm_list_12 = get_rm_data(answers_path_2, top_k_docs_path)

# all_rm_list = rm_list_11 + rm_list_12
all_rm_list = rm_list_12
print('len(all_rm_list): {}'.format(len(all_rm_list)))
save_results_path = "/root/paddlejob/workspace/env_run/rag/data"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/train_rm_pair_data.json', 'w') as file:
    json.dump(all_rm_list, file)
