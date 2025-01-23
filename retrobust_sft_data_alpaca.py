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

def get_positive_position(list_length=10, selected_number=3):
    # 初始化一个全为0的列表
    binary_list = [0] * list_length
    # 随机选择 selected_number 个不同的索引
    indices_to_set = random.sample(range(list_length), selected_number)
    # 将选定索引位置的元素置为1
    for index in indices_to_set:
        binary_list[index] = 1

    return binary_list

def remove_punctuation(text):
    # 定义要去除的标点符号
    punctuation = set('.,!?;:"()[]{}-')
    # 过滤掉标点符号
    return ''.join(char for char in text if char not in punctuation)

def clean_and_split(text):
    # 去除标点符号
    cleaned_text = remove_punctuation(text).lower()
    # 以空格为分隔符分割单词
    words = cleaned_text.split()
    return words

def calculate_match_ratio(answer, document, i):
    # 常见词汇列表，包括介词、冠词、代词、连词、助动词、否定词和疑问词
    common_words = {
        # 常见介词、冠词、代词、连词、助动词、否定词、疑问词
        "in", "on", "at", "to", "for", "with", "by", "from", "about",
        "a", "an", "the",
        "it", "they", "we", "you", "he", "she", "i", "me", "my", "mine", "ours", "us", "your", "yours", "his", "hers", "their", "theirs",
        "and", "or", "but", "because", "if", "then", "than", "as",
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "having", "be", "been", "being",
        "not", "no", "nor", "none",
        "what", "where", "when", "who", "why", "how", "which", "whom", "whose",
        # 常用标点符号
        ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "...", "--", "/", "\\", "|", "<", ">", "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
        # 其他常见停用词
        "of", "that", "this", "these", "those", "such", "there", "here", "all", "any", "both", "each", "few", "more", "some", "most", "other", "another", "every", "either", "neither"
    }

    # 清理并分割answer和document
    answer_words = [word for word in clean_and_split(answer) if word not in common_words and len(word) > 1]
    # document_words = set(word for word in clean_and_split(document) if word not in common_words)
    document_words = remove_punctuation(document).lower()
    # document_words_2 = clean_and_split(document)

    # if i < 100 and (extra == 'yes' or extra == 'no'):
    #     print(i, answer_words, document_words)
    #     print('\n')

    # 计算answer中有多少单词出现在document中
    match_count = sum(1 for word in answer_words if word in document_words)
    # match_count += sum(1 for word in answer_words if word in document_words_2)

    # 计算比例
    if len(answer_words) == 0:
        return 0.0  # 避免除以0
    match_ratio = match_count / (2*len(answer_words))

    return match_ratio

def sort_and_classify_documents(answer, documents, i):
    # 计算每个文档的匹配比例
    document_ratios = [(document, calculate_match_ratio(answer, document, i)) for document in documents]

    # 将文档划分为两个列表：匹配比例大于0和等于0
    positive_document_ratios = [doc_ratio for doc_ratio in document_ratios if doc_ratio[1] > 0]
    negative_document_ratios = [doc_ratio for doc_ratio in document_ratios if doc_ratio[1] == 0]

    # 对匹配比例大于0的文档进行排序
    positive_document_ratios.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的文档
    sorted_positive_documents = [doc for doc, _ in positive_document_ratios]
    sorted_negative_documents = [doc for doc, _ in negative_document_ratios]

    # 返回两个列表
    return sorted_positive_documents, sorted_negative_documents

dataset_name = '2wikimultihopqa'
top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/{}/retrobust_train_top_k_docs.jsonl'.format(dataset_name)

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

retrobust_sft_list = []

for i in tqdm(range(len(top_k_list))):
    cur_que_ans_docs = top_k_list[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_k_docs = cur_que_ans_docs['top_k_docs']
    all_documents_content = [item['content'] for item in top_k_docs]

    # ****************** generator ******************

    temp_dict = {}
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(all_documents_content)):
        doc_content = all_documents_content[doc_id]
        input_content = input_content + "Document{}: {}\n\n".format(str(doc_id), str(doc_content))

    temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."
    input_content = input_content + "Now, answer the Question: {}, based on the above Documents".format(str(question))
    temp_dict['input'] = input_content
    temp_dict['output'] = '**' + str(answer) + '**'
    temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."
    temp_dict['history'] = []

    retrobust_sft_list.append(temp_dict)


# 打乱列表
random.shuffle(retrobust_sft_list)

save_results_path = "/root/paddlejob/workspace/env_run/rag/data/{}".format(dataset_name)
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

# 保存答案为json
with open(save_results_path+'/retrobust_{}_sft_data.json'.format(dataset_name), 'w') as file:
    json.dump(retrobust_sft_list, file)


print('len(retrobust_sft_list): ', len(retrobust_sft_list))

print('Finished!')