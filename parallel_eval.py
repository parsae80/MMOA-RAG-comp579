import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from torch.nn.parallel import DataParallel
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time
import json
from normalize_answers import *
from collections import Counter
import os
import random
from transformers import LogitsProcessorList, LogitsProcessor
from limited_tokens import get_allowed_tokens, AllowedTokensLogitsProcessor


def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def convert_to_int_list(s, K_candidate):
    int_list = []
    for x in s.split(','):
        try:
            # 尝试转换为整数
            num = int(x.strip())
            if num < 0 or num >= K_candidate:
                num = random.randint(0, K_candidate-1)
                print('模型输出的文档ID为 {} , 这不在符合规定的范围内, 重新随机赋值为: {}\n'.format(int(x.strip()), num))
            int_list.append(num)
        except ValueError:
            num = random.randint(0, K_candidate-1)
            print('转换原始字符 {} 失败, 随机赋值为: {}'.format(x, num))
            print('对应完整的原始answers为:', s, '\n')
            continue
    return int_list
    
def extract_digits(input_string):
    input_string = input_string.replace(",", "")
    # 创建一个空列表来存储结果
    digits_list = []
    
    # 遍历输入字符串中的每个字符
    for char in input_string:
        # 检查字符是否为数字
        if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # 将字符转换为整数并添加到列表中
            digits_list.append(int(char))

    # 去重复
    my_list = digits_list
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)
    
    return unique_list

def compute_scores(predict_answers, golden_answers):
    assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
    final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
    total = len(predict_answers)

    for prediction, ground_truth in zip(predict_answers, golden_answers):
        normalized_prediction = normalize_answer_final(prediction)
        normalized_ground_truth = normalize_answer_final(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
            final_metric["acc"] += 1.0

        if normalized_prediction == normalized_ground_truth:
            final_metric["em"] += 1.0

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["f1"] += f1
        final_metric["precision"] += precision
        final_metric["recall"] += recall

    for k in ['acc', 'em', 'f1', 'precision', 'recall']:
        final_metric[k] /= total

    return final_metric

def get_response(messages):
    if tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).cuda()
    else:
        conv = get_conversation_template(generator_model_path)
        for message in messages:
            conv.append_message(message["role"], message["content"])
        conv.append_message("assistant", "")
        input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # prefix_allowed_tokens_fn=lambda batch_id, prev_ids: allowed_tokens,
    )
    outputs = tokenizer.decode(
        outputs[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return outputs

def batch_get_response(model, tokenizer, messages_list, logits_processor=[]):
    # Create a list of input_ids for all messages in the batch
    input_ids_list = []
    for messages in messages_list:
        if tokenizer.chat_template is not None:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to("cuda:0")
        else:
            conv = get_conversation_template(generator_model_path)
            for message in messages:
                conv.append_message(message["role"], message["content"])
            conv.append_message("assistant", "")
            input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.to("cuda:0")
        
        input_ids_list.append(input_ids)

    # Pad input_ids to the same length and create a single tensor
    input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]  # 向量需要是一维的

    # 找出最长序列的长度
    max_length = max(input_ids.size(0) for input_ids in input_ids_list)
    # 手动左填充
    input_ids_padded = torch.stack([
        torch.cat([input_ids.new_full((max_length - input_ids.size(0),), tokenizer.eos_token_id), input_ids], dim=0)
        for input_ids in input_ids_list
    ], dim=0)

    # 创建 attention mask
    attention_masks = torch.stack([
        torch.cat([torch.zeros(max_length - input_ids.size(0), dtype=torch.long), torch.ones(input_ids.size(0), dtype=torch.long)], dim=0)
        for input_ids in input_ids_list
    ], dim=0).to("cuda:0")

    # Generate outputs for all inputs in the batch
    if len(logits_processor) != 0:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=19,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode each generated output separately
    results = []
    for i in range(outputs.shape[0]):
        # 因为是左填充，所以生成的输出部分从填充长度位置开始
        output = tokenizer.decode(
            outputs[i, input_ids_padded[i].size(0):],
            skip_special_tokens=True,
        )
        results.append(output)

    return results

def get_selector_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))

    message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(top_k_docs)-1)}, 
        {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."}
    ]

    return message

def get_generator_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n".format(str(doc_id), str(doc_content))

    if len(top_k_docs) > 0:
        input_content = input_content + "\nNow, answer the Question: {}, based on the above Documents".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."}
    ]

    elif len(top_k_docs) == 0:
        input_content = input_content + "\nNow, answer the Question: {}.".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question."}
    ]

    return message

def get_selector_post_role_prompt(question, top_k_docs):

    return {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: i,j,k,l (i,j,k,l represent the concrete Documents ID 0,1,2,... {}). Do not output duplicate Document ID.".format(len(top_k_docs)-1, str(question), len(top_k_docs)-1)}

def get_generator_post_role_prompt(top_docs):

    if len(top_docs) > 0:
        message = {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}
    elif len(top_docs) == 0:
        message = {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer** and nothing else."}

    return message

def get_messages(question, top_docs):
    messages = get_generator_prefix_role_prompt(question, top_docs)
    messages.append(get_generator_post_role_prompt(top_docs))

    return messages

def get_selector_messages(question, top_docs):
    messages = get_selector_prefix_role_prompt(question, top_docs)
    messages.append(get_selector_post_role_prompt(question, top_docs))

    return messages

top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/hotpotqa/val_top_k_docs.jsonl'
generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/hotpotqa/selector_and_generator/ppo_2_04'
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'
K_candidate = 10

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

# loading generator model
print('*'*20)
print('loading generator model')
model = AutoModelForCausalLM.from_pretrained(
    generator_model_path,
    torch_dtype="auto",
).to("cuda:0")  # 将模型加载到第一个GPU上
model = DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]) # 使用DataParallel进行并行预测
model.eval()  # 切换到评估模式
tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading selector model
model_selector = model
tokenizer_selector = tokenizer

# random.shuffle(top_k_list)
# top_k_list = top_k_list[:3000]

all_questions = []
all_answers = []
all_top_docs = []
for i in range(len(top_k_list)):
    cur_que_ans_docs = top_k_list[i]
    all_questions.append(cur_que_ans_docs['question'])
    all_answers.append(cur_que_ans_docs['answer'])
    all_top_docs.append([doc['content'] for doc in cur_que_ans_docs['top_k_docs']])

shuffled_all_top_docs = all_top_docs[:]
for i in range(len(shuffled_all_top_docs)):
    temp_list = shuffled_all_top_docs[i][:]
    random.shuffle(temp_list)
    shuffled_all_top_docs[i] = temp_list

# answers
selector_answers = []
generator_answers = []

# get selector answers list
selector_messages_list = []
for i in range(len(top_k_list)):
    question = all_questions[i]
    top_docs = shuffled_all_top_docs[i]
    selector_messages = get_selector_messages(question, top_docs)
    selector_messages_list.append(selector_messages)
print("len(selector_messages_list): {}".format(len(selector_messages_list)))

# *******************************selector*******************************
batch_size = 8
print('*'*20)
print('selector generating.')
# 定义允许的token
allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',']
allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
eos_token_id = tokenizer.eos_token_id
allowed_token_ids.append(eos_token_id)
# 创建LogitsProcessor
logits_processor = LogitsProcessorList([
    AllowedTokensLogitsProcessor(allowed_token_ids)
])
for i in tqdm(range(0, len(selector_messages_list), batch_size)):
    messages_batch = selector_messages_list[i: i+batch_size]
    answers_batch = batch_get_response(model_selector, tokenizer_selector, messages_batch, logits_processor)
    for answer in answers_batch:
        number_answer = extract_digits(answer)
        selector_answers.append(number_answer)
        # selector_answers.append([0,1,2,3,4])

        with open("./sft_1epoch_selector.txt", "a", encoding="utf-8") as file:
            file.write(answer + '\n')

        if i < 20:
            print(answer, number_answer)

# *******************************generating*******************************
print('*'*20)
print('generating')
start_time = time.time()

generator_messages_list = []
for i in range(len(top_k_list)):
    question = all_questions[i]
    top_docs = shuffled_all_top_docs[i]
    selected_ids = selector_answers[i]
    generator_input_docs = []
    for selected_id in selected_ids:
        generator_input_docs.append(top_docs[selected_id])
    # print(i, len(generator_input_docs))
    messages = get_messages(question=question, top_docs=generator_input_docs)
    generator_messages_list.append(messages)
print("len(generator_messages_list): {}".format(len(generator_messages_list)))

batch_size = 8
print('*'*20)
print('generator generating.')
for i in tqdm(range(0, len(generator_messages_list), batch_size)):
    messages_batch = generator_messages_list[i: i+batch_size]
    answers_batch = answers_batch = batch_get_response(model_selector, tokenizer_selector, messages_batch)
    # print(answers_batch)
    for answer in answers_batch:
        answer = answer.replace("*", "")
        generator_answers.append(answer)

        with open("./sft_1epoch_generator.txt", "a", encoding="utf-8") as file:
            file.write(answer + '\n')

        if i < 20:
            print(answer)

# getting F1 score
print(compute_scores(generator_answers, all_answers))

save_results_path = "/root/paddlejob/workspace/env_run/rag/data/hotpotqa"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/ppo_2_04epoch_val.jsonl', 'w') as file: # 
    for i in range(len(generator_answers)):
        temp_dic = {'question_id': str(i), 'question': str(all_questions[i]), 'predict_answer': str(generator_answers[i]), 'golden_answer': str(all_answers[i])}
        file.write(json.dumps(temp_dic) + '\n')

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

print('Finished!')